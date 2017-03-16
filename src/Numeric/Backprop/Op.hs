{-# LANGUAGE AllowAmbiguousTypes  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE LambdaCase           #-}
{-# LANGUAGE PatternSynonyms      #-}
{-# LANGUAGE PolyKinds            #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns         #-}

-- |
-- Module      : Numeric.Backprop.Op
-- Copyright   : (c) Justin Le 2017
-- License     : BSD3
--
-- Maintainer  : justin@jle.im
-- Stability   : experimental
-- Portability : non-portable
--
-- Provides the 'Op' (and 'OpM') type and combinators, which represent
-- differentiable functions/operations on values, and are used by the
-- library to perform back-propagation.
--
-- Note that 'Op' is a /subset/ or /subtype/ of 'OpM', and so, any function
-- that expects an @'OpM' m as a@ (or an @'Numeric.Backprop.OpB' s as a@)
-- can be given an @'Op' as a@ and it'll work just fine.
--

module Numeric.Backprop.Op (
  -- * Implementation
  -- $opdoc
  -- * Types
  -- ** Op and Synonyms
    Op, pattern Op, OpM(..)
  -- ** Tuple Types
  -- | See "Numeric.Backprop#prod" for a mini-tutorial on 'Prod' and
  -- 'Tuple'
  , Prod(..), Tuple, I(..)
  -- * Running
  -- ** Pure
  , runOp, gradOp, gradOp', gradOpWith, gradOpWith', runOp'
  -- ** Monadic
  , runOpM, gradOpM, gradOpM', gradOpWithM, gradOpWithM', runOpM'
  -- * Manipulation
  , composeOp, (~.)
  -- * Creation
  , op0, opConst
  , opConst'
  -- ** Automatic creation using the /ad/ library
  , op1, op2, op3, opN
  , Replicate
  -- ** Giving gradients directly
  , op1', op2', op3'
  -- ** From Isomorphisms
  , opCoerce, opIso
  -- * Utility
  , pattern (:>), only, head'
  , pattern (::<), only_
  -- ** Numeric Ops#numops#
  -- $numops
  , (+.), (-.), (*.), negateOp, absOp, signumOp
  , (/.), recipOp
  , expOp, logOp, sqrtOp, (**.), logBaseOp
  , sinOp, cosOp, tanOp, asinOp, acosOp, atanOp
  , sinhOp, coshOp, tanhOp, asinhOp, acoshOp, atanhOp
  ) where

import           Data.Bifunctor
import           Data.Coerce
import           Data.Maybe
import           Data.Reflection                (Reifies)
import           Data.Type.Combinator
import           Data.Type.Conjunction
import           Data.Type.Index
import           Data.Type.Length
import           Data.Type.Nat
import           Data.Type.Product
import           Data.Type.Util
import           Data.Type.Vector hiding        (head')
import           Lens.Micro.Extras
import           Numeric.AD
import           Numeric.AD.Internal.Reverse    (Reverse, Tape)
import           Numeric.AD.Mode.Forward hiding (grad')
import           Numeric.Backprop.Iso
import           Type.Class.Higher
import           Type.Class.Known
import           Type.Class.Witness
import           Type.Family.List hiding        (Reverse)

-- instead of Tuple as, Prod Diff as, where Diff can be a value, or zero,
-- or one?

-- $opdoc
-- 'Op's contain information on a function as well as its gradient, but
-- provides that information in a way that allows them to be "chained".
--
-- For example, for a function
--
-- \[
-- f : \mathbb{R}^n \rightarrow \mathbb{R}
-- \]
--
-- We might want to apply a function \(g\) to the result we get, to get
-- our "final" result:
--
-- \[
-- \eqalign{
-- y &= f(\mathbf{x})\cr
-- z &= g(y)
-- }
-- \]
--
-- Now, we might want the gradient \(\nabla z\) with respect to
-- \(\mathbf{x}\), or \(\nabla_\mathbf{x} z\).  Explicitly, this is:
--
-- \[
-- \nabla_\mathbf{x} z = \left< \frac{\partial z}{\partial x_1}, \frac{\partial z}{\partial x_2}, \ldots \right>
-- \]
--
-- We can compute that by multiplying the total derivative of \(z\) with
-- respect to \(y\) (that is, \(\frac{dz}{dy}\)) with the gradient of
-- \(f\)) itself:
--
-- \[
-- \eqalign{
-- \nabla_\mathbf{x} z &= \frac{dz}{dy} \left< \frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2}, \ldots \right>\cr
-- \nabla_\mathbf{x} z &= \frac{dz}{dy} \nabla_\mathbf{x} y
-- }
-- \]
--
-- So, to create an @'Op' as a@ with the 'Op' constructor (or an 'OpM' with the
-- 'OpM' constructor), you give a function that returns a tuple,
-- containing:
--
--     1. An @a@: The result of the function
--     2. An @Maybe a -> Tuple as@:  A function that, when given
--     \(\frac{dz}{dy}\) (in a 'Just'), returns the total gradient
--     \(\nabla_z \mathbf{x}\).  If the function is given is given
--     'Nothing', then \(\frac{dz}{dy}\) should be taken to be 1.  In other
--     words, you would simply need to return \(\nabla_y \mathbf{x}\),
--     unchanged.  That is, an input of 'Nothing' indicates that the "final
--     result" is just simply \(f(\mathbf{x})\), and not some
--     \(g(f(\mathbf{x}))\).
--
-- This is done so that 'Op's can easily be "chained" together, one after
-- the other.  If you have an 'Op' for \(f\) and an 'Op' for \(g\), you can
-- compute the gradient of \(f\) knowing that the result target is
-- \(g \circ f\).
--
-- Note that end users should probably never be required to construct an
-- 'Op' or 'OpM' explicitly this way.  Instead, libraries should provide
-- carefuly pre-constructed ones, or provide ways to generate them
-- automatically (like 'op1', 'op2', and 'op3' here).
--
-- For examples of 'Op's implemented from scratch, see the implementations
-- of '+.', '-.', 'recipOp', 'sinOp', etc.

-- | An @'OpM' m as a@ represents a /differentiable/ (monadic) function
-- from @as@ to @a@, in the context of a 'Monad' @m@.
--
-- For example, an
--
-- @
-- 'OpM' IO '[Int, Bool] Double
-- @
--
-- would be a function that takes an 'Int' and a 'Bool' and returns
-- a 'Double' (in 'IO').  It can be differentiated to give a /gradient/ of
-- an 'Int' and a 'Bool' (also in 'IO') if given the total derivative for
-- the @Double@.
--
-- Note that an 'OpM' is a /superclass/ of 'Op', so any function that
-- expects an @'OpM' m as a@ can also accept an @'Op' as a@.
--
-- See 'runOpM', 'gradOpM', and 'gradOpWithM' for examples on how to run
-- it.
newtype OpM m as bs =
    -- | Construct an 'OpM' by giving a (monadic) function creating the
    -- result, and also a continuation on how to create the gradient, given
    -- the total derivative of @a@.
    --
    -- See the module documentation for "Numeric.Backprop.Op" for more
    -- details on the function that this constructor and 'Op' expect.
    OpM (Tuple as -> m (Tuple bs, Prod Maybe bs -> m (Tuple as)))

-- | An @'Op' as a@ describes a differentiable function from @as@ to @a@.
--
-- For example, a value of type
--
-- @
-- 'Op' '[Int, Bool] Double
-- @
--
-- is a function from an 'Int' and a 'Bool', returning a 'Double'.  It can
-- be differentiated to give a /gradient/ of an 'Int' and a 'Bool' if given
-- a total derivative for the @Double@.  If we call 'Bool' \(2\), then,
-- mathematically, it is akin to a:
--
-- \[
-- f : \mathbb{Z} \times 2 \rightarrow \mathbb{R}
-- \]
--
-- See 'runOp', 'gradOp', and 'gradOpWith' for examples on how to run it,
-- and 'Op' for instructions on creating it.
--
-- This type is abstracted over using the pattern synonym with constructor
-- 'Op', so you can create one from scratch with it.  However, it's
-- simplest to create it using 'op2'', 'op1'', 'op2'', and 'op3'' helper
-- smart constructors  And, if your function is a numeric function, they
-- can even be created automatically using 'op1', 'op2', 'op3', and 'opN'
-- with a little help from "Numeric.AD" from the /ad/ library.
--
-- Note that this type is a /subset/ or /subtype/ of 'OpM' (and also of
-- 'Numeric.Backprop.OpB').  So, if a function ever expects an @'OpM' m as
-- a@ (or a 'Numeric.Backprop.OpB'), you can always provide an @'Op' as a@
-- instead.
--
-- Many functions in this library will expect an @'OpM' m as a@ (or
-- an @'Numeric.Backprop.OpB' s as a@), and in all of these cases, you can
-- provide an @'Op' as a@.
type Op as bs = forall m. Monad m => OpM m as bs

-- | Construct an 'Op' by giving a function creating the result, and also
-- a continuation on how to create the gradient, given the total derivative
-- of @a@.
--
-- See the module documentation for "Numeric.Backprop.Op" for more details
-- on the function that this constructor and 'OpM' expect.
pattern Op :: (Tuple as -> (Tuple bs, Prod Maybe bs -> Tuple as)) -> Op as bs
pattern Op runOp' <- OpM (\f -> (second . fmap) getI . getI . f -> runOp')
  where
    Op f = OpM (pure . (second . fmap) pure . f)

-- | A combination of 'runOpM' and 'gradOpWithM''.  Given an 'OpM' and
-- inputs, returns the result of the 'OpM' and a continuation that gives
-- its gradient.
--
-- The continuation takes the total derivative of the result as input.  See
-- documenation for 'gradOpWithM'' and module documentation for
-- "Numeric.Backprop.Op" for more information.
runOpM'
    :: OpM m as bs
    -> Tuple as
    -> m (Tuple bs, Prod Maybe bs -> m (Tuple as))
runOpM' (OpM f) = f

-- | A combination of 'runOp' and 'gradOpWith''.  Given an 'Op' and inputs,
-- returns the result of the 'Op' and a continuation that gives its
-- gradient.
--
-- The continuation takes the total derivative of the result as input.  See
-- documenation for 'gradOpWith'' and module documentation for
-- "Numeric.Backprop.Op" for more information.
runOp'
    :: Op as bs
    -> Tuple as
    -> (Tuple bs, Prod Maybe bs -> Tuple as)
runOp' o = (second . fmap) getI . getI . runOpM' o

composeOp
    :: forall m as bs cs. Monad m
    => OpM m as bs
    -> OpM m bs cs
    -> OpM m as cs
composeOp f g = OpM $ \xs -> do
    (ys, gF) <- runOpM' f xs
    (zs, gG) <- runOpM' g ys
    let gH :: Prod Maybe cs -> m (Tuple as)
        gH dzs = do
          dys <- gG dzs
          gF (map1 (Just . getI) dys)
    return (zs, gH)

(~.)
    :: forall m as bs cs. Monad m
    => OpM m bs cs
    -> OpM m as bs
    -> OpM m as cs
f ~. g = composeOp g f

zipOp'
    :: forall m as bss. (Monad m, Every Num as, Known Length as)
    => Prod Length bss
    -> Prod (OpM m as) bss
    -> OpM m as (Concat bss)
zipOp' = \case
    Ø -> \case
      Ø -> OpM $ \_ ->
        return (Ø, \case Ø -> return (map1 ((0 \\) . every @_ @Num) indices))
    (l :: Length cs) :< (ls :: Prod Length css) -> \case
      o :< os -> OpM $ \xs -> do
        (ys , f ) <- runOpM' o xs
        (yss, fs) <- runOpM' (zipOp' ls os) xs
        let zs :: Tuple (Concat bss)
            zs = ys `append'` yss
            gs :: Prod Maybe (Concat bss) -> m (Tuple as)
            gs dzss = case splitAt' @_ @cs @(Concat css) l dzss of
              (dys, dyss) -> do
                dx  <- f  dys
                dxs <- fs dyss
                return $ imap1 (\ix (I d1 :&: I d2) -> I (d1 + d2)
                                    \\ every @_ @Num ix
                               ) (dx `zipP` dxs)
        return (zs, gs)

zipOp
    :: forall m as bss. (Monad m, Every Num as, Known Length as, Known Length bss, Every (Known Length) bss)
    => Prod (OpM m as) bss
    -> OpM m as (Concat bss)
zipOp = zipOp' known

-- | Run the function that an 'Op' encodes, to get the result.
--
-- >>> runOp (op2 (*)) (3 ::< 5 ::< Ø)
-- 15
runOp :: Op as bs -> Tuple as -> Tuple bs
runOp o = fst . runOp' o

-- | Run the function that an 'Op' encodes, to get the resulting output and
-- also its gradient with respect to the inputs.
--
-- >>> gradOpM' (op2 (*)) (3 ::< 5 ::< Ø) :: IO (Int, Tuple '[Int, Int])
-- (15, 5 ::< 3 ::< Ø)
gradOp' :: Known Length bs => Op as bs -> Tuple as -> (Tuple bs, Tuple as)
gradOp' o = second ($ lengthProd Nothing known) . runOp' o

-- | The monadic version of 'runOp', for 'OpM's.
--
-- >>> runOpM (op2 (*)) (3 ::< 5 ::< Ø) :: IO Int
-- 15
runOpM :: Functor m => OpM m as bs -> Tuple as -> m (Tuple bs)
runOpM o = fmap fst . runOpM' o

-- | The monadic version of 'gradOp'', for 'OpM's.
gradOpM' :: (Monad m, Known Length bs) => OpM m as bs -> Tuple as -> m (Tuple bs, Tuple as)
gradOpM' o x = do
    (y, gF) <- runOpM' o x
    g <- gF (lengthProd Nothing known)
    return (y, g)

-- | A combination of 'gradOp' and 'gradOpWith'.  The third argument is
-- (optionally) the total derivative the result.  Give 'Nothing' and it is
-- assumed that the result is the final result (and the total derivative is
-- 1), and this behaves the same as 'gradOp'.  Give @'Just' d@ and it uses
-- the @d@ as the total derivative of the result, and this behaves like
-- 'gradOpWith'.
--
-- See 'gradOp' and the module documentaiton for "Numeric.Backprop.Op" for
-- more information.
gradOpWith'
    :: Op as bs         -- ^ 'Op' to run
    -> Tuple as         -- ^ Inputs to run it with
    -> Prod Maybe bs    -- ^ If 'Just', taken as the total derivative of
                        --     the result.  If 'Nothing', assumes that the
                        --     result is the final result.
    -> Tuple as         -- ^ The gradient
gradOpWith' o = snd . runOp' o

-- | The monadic version of 'gradOpWith'', for 'OpM's.
gradOpWithM'
    :: Monad m
    => OpM m as bs      -- ^ 'OpM' to run
    -> Tuple as         -- ^ Inputs to run it with
    -> Prod Maybe bs    -- ^ If 'Just', taken as the total derivative of the
                        --     result.  If 'Nothing', assumes that the result is
                        --     the final result.
    -> m (Tuple as)     -- ^ The gradient
gradOpWithM' o xs g = do
    (_, f) <- runOpM' o xs
    f g

-- | Run the function that an 'Op' encodes, and get the gradient of
-- a "final result" with respect to the inputs, given the total derivative
-- of the output with the final result.
--
-- See 'gradOp' and the module documentaiton for "Numeric.Backprop.Op" for
-- more information.
gradOpWith
    :: Op as bs     -- ^ 'Op' to run
    -> Tuple as     -- ^ Inputs to run it with
    -> Tuple bs     -- ^ The total derivative of the result
    -> Tuple as     -- ^ The gradient
gradOpWith o i = gradOpWith' o i . map1 (Just . getI)

-- | The monadic version of 'gradOpWith', for 'OpM's.
gradOpWithM
    :: Monad m
    => OpM m as bs      -- ^ 'OpM' to run
    -> Tuple as         -- ^ Inputs to run it with
    -> Tuple bs         -- ^ The total derivative of the result
    -> m (Tuple as)     -- ^ the gradient
gradOpWithM o i = gradOpWithM' o i . map1 (Just . getI)

-- | Run the function that an 'Op' encodes, and get the gradient of the
-- output with respect to the inputs.
--
-- >>> gradOp (op2 (*)) (3 ::< 5 ::< Ø)
-- 5 ::< 3 ::< Ø
-- -- the gradient of x*y is (y, x)
gradOp :: Known Length bs => Op as bs -> Tuple as -> Tuple as
gradOp o i = gradOpWith' o i (lengthProd Nothing known)

-- | The monadic version of 'gradOp', for 'OpM's.
gradOpM :: (Known Length bs, Monad m) => OpM m as bs -> Tuple as -> m (Tuple as)
gradOpM o i = do
    (_, gF) <- runOpM' o i
    gF (lengthProd Nothing known)

-- | An 'Op' that coerces an item into another item whose type has the same
-- runtime representation.  Requires the input to be an instance of 'Num'.
--
-- >>> gradOp' opCoerce (Identity 5) :: (Int, Identity Int)
-- (5, Identity 1)
--
-- @
-- 'opCoerce' = 'opIso' 'coerced'
-- @
opCoerce :: Num a => Coercible a b => Op '[a] '[b]
opCoerce = opIso coerced

-- | An 'Op' that runs the input value through the isomorphism encoded in
-- the 'Iso'.  Requires the input to be an instance of 'Num'.
--
-- Warning: This is unsafe!  It assumes that the isomorphisms themselves
-- have derivative 1, so will break for things like
-- 'Numeric.Lens.exponentiating'.  Basically, don't use this for any
-- "numeric" isomorphisms.
opIso :: Num a => Iso' a b -> Op '[a] '[b]
opIso i = op1' $ \x -> (only_ (view i x), maybe 1 (review i) . head')

-- | A version of 'opConst' taking explicit 'Length', indicating the
-- number of inputs and their types.
--
-- Requiring an explicit 'Length' is mostly useful for rare "extremely
-- polymorphic" situations, where GHC can't infer the type and length of
-- the the expected input tuple.  If you ever actually explicitly write
-- down @as@ as a list of types, you should be able to just use
-- 'opConst'.
opConst' :: forall as bs. Every Num as => Length as -> Tuple bs -> Op as bs
opConst' l x = Op $ \_ ->
    (x , const $ map1 ((0 \\) . every @_ @Num) (indices' l))

-- | An 'Op' that ignores all of its inputs and returns a given constant
-- value.
--
-- >>> gradOp' (opConst 10) (1 ::< 2 ::< 3 ::< Ø)
-- (10, 0 ::< 0 ::< 0 ::< Ø)
opConst :: forall as bs. (Every Num as, Known Length as) => Tuple bs -> Op as bs
opConst = opConst' known

-- | Create an 'Op' that takes no inputs and always returns the given
-- value.
--
-- There is no gradient, of course (using 'gradOp' will give you an empty
-- tuple), because there is no input to have a gradient of.
--
-- >>> gradOp' (op0 10) Ø
-- (10, Ø)
--
-- For a constant 'Op' that takes input and ignores it, see 'opConst' and
-- 'opConst''.
--
-- Note that because this returns an 'Op', it can be used with any function
-- that expects an 'OpM' or 'Numeric.Backprop.OpB', as well.
op0 :: Tuple as -> Op '[] as
op0 x = Op $ \case
    Ø -> (x, const Ø)
{-# INLINE op0 #-}

-- | Create an 'Op' of a function taking one input, by giving its explicit
-- derivative.  The function should return a tuple containing the result of
-- the function, and also a function taking the derivative of the result
-- and return the derivative of the input.
--
-- If we have
--
-- \[
-- \eqalign{
-- f &: \mathbb{R} \rightarrow \mathbb{R}\cr
-- y &= f(x)\cr
-- z &= g(y)
-- }
-- \]
--
-- Then the derivative \( \frac{dz}{dx} \), it would be:
--
-- \[
-- \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}
-- \]
--
-- If our 'Op' represents \(f\), then the second item in the resulting
-- tuple should be a function that takes \(\frac{dz}{dy}\) and returns
-- \(\frac{dz}{dx}\).
--
-- If the input is 'Nothing', then \(\frac{dz}{dy}\) should be taken to be
-- \(1\).
--
-- As an example, here is an 'Op' that squares its input:
--
-- @
-- square :: Num a => 'Op' '[a] a
-- square = 'op1'' $ \\x -> (x*x, \\case Nothing -> 2 * x
--                                   Just d  -> 2 * d * x
--                       )
-- @
--
-- Remember that, generally, end users shouldn't directly construct 'Op's;
-- they should be provided by libraries or generated automatically.
--
-- For numeric functions, single-input 'Op's can be generated automatically
-- using 'op1'.
op1'
    :: (a -> (Tuple bs, Prod Maybe bs -> a))
    -> Op '[a] bs
op1' f = Op $ \case
    I x :< Ø ->
      let (y, dx) = f x
      in  (y, only_ . dx)
{-# INLINE op1' #-}

-- | Create an 'Op' of a function taking two inputs, by giving its explicit
-- gradient.  The function should return a tuple containing the result of
-- the function, and also a function taking the derivative of the result
-- and return the derivative of the input.
--
-- If we have
--
-- \[
-- \eqalign{
-- f &: \mathbb{R}^2 \rightarrow \mathbb{R}\cr
-- z &= f(x, y)\cr
-- k &= g(z)
-- }
-- \]
--
-- Then the gradient \( \left< \frac{\partial k}{\partial x}, \frac{\partial k}{\partial y} \right> \)
-- would be:
--
-- \[
-- \left< \frac{\partial k}{\partial x}, \frac{\partial k}{\partial y} \right> =
--  \left< \frac{dk}{dz} \frac{\partial z}{dx}, \frac{dk}{dz} \frac{\partial z}{dy} \right>
-- \]
--
-- If our 'Op' represents \(f\), then the second item in the resulting
-- tuple should be a function that takes \(\frac{dk}{dz}\) and returns
-- \( \left< \frac{\partial k}{dx}, \frac{\partial k}{dx} \right> \).
--
-- If the input is 'Nothing', then \(\frac{dk}{dz}\) should be taken to be
-- \(1\).
--
-- As an example, here is an 'Op' that multiplies its inputs:
--
-- @
-- mul :: Num a => 'Op' '[a, a] a
-- mul = 'op2'' $ \\x y -> (x*y, \\case Nothing -> (y  , x  )
--                                  Just d  -> (d*y, x*d)
--                      )
-- @
--
-- Remember that, generally, end users shouldn't directly construct 'Op's;
-- they should be provided by libraries or generated automatically.
--
-- For numeric functions, two-input 'Op's can be generated automatically
-- using 'op2'.
op2'
    :: (a -> b -> (Tuple cs, Prod Maybe cs -> (a, b)))
    -> Op '[a,b] cs
op2' f = Op $ \case
    I x :< I y :< Ø ->
      let (z, dxdy) = f x y
      in  (z, (\(dx,dy) -> dx ::< dy ::< Ø) . dxdy)
{-# INLINE op2' #-}

-- | Create an 'Op' of a function taking three inputs, by giving its explicit
-- gradient.  See documentation for 'op2'' for more details.
op3'
    :: (a -> b -> c -> (Tuple ds, Prod Maybe ds -> (a, b, c)))
    -> Op '[a,b,c] ds
op3' f = Op $ \case
    I x :< I y :< I z :< Ø ->
      let (q, dxdydz) = f x y z
      in  (q, (\(dx, dy, dz) -> dx ::< dy ::< dz ::< Ø) . dxdydz)
{-# INLINE op3' #-}

-- | Automatically create an 'Op' of a numerical function taking one
-- argument.  Uses 'Numeric.AD.diff', and so can take any numerical
-- function polymorphic over the standard numeric types.
--
-- >>> gradOp' (op1 (recip . negate)) (5 ::< Ø)
-- (-0.2, 0.04 ::< Ø)
op1 :: Num a
    => (forall s. AD s (Forward a) -> AD s (Forward a))
    -> Op '[a] '[a]
op1 f = op1' $ \x ->
    let (z, dx) = diff' f x
    in  (only_ z, maybe dx (* dx) . head')

-- | Automatically create an 'Op' of a numerical function taking two
-- arguments.  Uses 'Numeric.AD.grad', and so can take any numerical function
-- polymorphic over the standard numeric types.
--
-- >>> gradOp' (op2 (\x y -> x * sqrt y)) (3 ::< 4 ::< Ø)
-- (6.0, 2.0 ::< 0.75 ::< Ø)
op2 :: Num a
    => (forall s. Reifies s Tape => Reverse s a -> Reverse s a -> Reverse s a)
    -> Op '[a,a] '[a]
op2 f = opN $ \case I x :* I y :* ØV -> f x y

-- | Automatically create an 'Op' of a numerical function taking three
-- arguments.  Uses 'Numeric.AD.grad', and so can take any numerical function
-- polymorphic over the standard numeric types.
--
-- >>> gradOp' (op3 (\x y z -> (x * sqrt y)**z)) (3 ::< 4 ::< 2 ::< Ø)
-- (36.0, 24.0 ::< 9.0 ::< 64.503 ::< Ø)
op3 :: Num a
    => (forall s. Reifies s Tape => Reverse s a -> Reverse s a -> Reverse s a -> Reverse s a)
    -> Op '[a,a,a] '[a]
op3 f = opN $ \case I x :* I y :* I z :* ØV -> f x y z

-- | Automatically create an 'Op' of a numerical function taking multiple
-- arguments.  Uses 'Numeric.AD.grad', and so can take any numerical
-- function polymorphic over the standard numeric types.
--
-- >>> gradOp' (opN (\(x :+ y :+ Ø) -> x * sqrt y)) (3 ::< 4 ::< Ø)
-- (6.0, 2.0 ::< 0.75 ::< Ø)
opN :: (Num a, Known Nat n)
    => (forall s. Reifies s Tape => Vec n (Reverse s a) -> Reverse s a)
    -> Op (Replicate n a) '[a]
opN f = Op $ \xs ->
    let (y, dxs) = grad' f (prodToVec' known xs)
    in  (only_ y, vecToProd . maybe dxs (\q -> (q *) <$> dxs) . head')

instance (Monad m, Known Length as, Every Num as, Num a) => Num (OpM m as '[a]) where
    o1 + o2       = (+.)     ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE (+) #-}
    o1 - o2       = (-.)     ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE (-) #-}
    o1 * o2       = (*.)     ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE (*) #-}
    negate o      = negateOp ~. o
    {-# INLINE negate #-}
    signum o      = signumOp ~. o
    {-# INLINE signum #-}
    abs    o      = absOp    ~. o
    {-# INLINE abs #-}
    fromInteger x = opConst (only_ (fromInteger x))
    {-# INLINE fromInteger #-}

instance (Monad m, Known Length as, Every Fractional as, Every Num as, Fractional a) => Fractional (OpM m as '[a]) where
    o1 / o2        = (/.)    ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE (/) #-}
    recip o        = recipOp ~. o
    {-# INLINE recip #-}
    fromRational x = opConst (only_ (fromRational x))
    {-# INLINE fromRational #-}

instance (Monad m, Known Length as, Every Floating as, Every Fractional as, Every Num as, Floating a) => Floating (OpM m as '[a]) where
    pi            = opConst (only_ pi)
    {-# INLINE pi #-}
    exp   o       = expOp     ~. o
    {-# INLINE exp #-}
    log   o       = logOp     ~. o
    {-# INLINE log #-}
    sqrt  o       = sqrtOp    ~. o
    {-# INLINE sqrt #-}
    o1 ** o2      = (**.)     ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE (**) #-}
    logBase o1 o2 = logBaseOp ~. zipOp (o1 :< o2 :< Ø)
    {-# INLINE logBase #-}
    sin   o       = sinOp     ~. o
    {-# INLINE sin #-}
    cos   o       = cosOp     ~. o
    {-# INLINE cos #-}
    tan   o       = tanOp     ~. o
    {-# INLINE tan #-}
    asin  o       = asinOp    ~. o
    {-# INLINE asin #-}
    acos  o       = acosOp    ~. o
    {-# INLINE acos #-}
    atan  o       = atanOp    ~. o
    {-# INLINE atan #-}
    sinh  o       = sinhOp    ~. o
    {-# INLINE sinh #-}
    cosh  o       = coshOp    ~. o
    {-# INLINE cosh #-}
    tanh  o       = tanhOp    ~. o
    {-# INLINE tanh #-}
    asinh o       = asinhOp   ~. o
    {-# INLINE asinh #-}
    acosh o       = acoshOp   ~. o
    {-# INLINE acosh #-}
    atanh o       = atanhOp   ~. o
    {-# INLINE atanh #-}

-- $numops
--
-- Built-in ops for common numeric operations, implemented directly so
-- that they are more efficient than using 'op1' \/ 'op2' etc.
--
-- The naming scheme is:
--
-- @
-- ('+.') = 'op2' ('+')
-- 'negateOp' = 'op1' 'negate
-- @
--
-- Note that the operators (like '+.') are meant to be used in prefix
-- form, like:
--
-- @
-- 'Numeric.Backprop.liftB2' ('.+') v1 v2
-- @

-- | Optimized version of @'op1' ('+')@.
(+.) :: Num a => Op '[a, a] '[a]
(+.) = op2' $ \x y -> (only_ (x + y), maybe (1, 1) (\g -> (g, g)) . head')
{-# INLINE (+.) #-}

-- | Optimized version of @'op1' ('-')@.
(-.) :: Num a => Op '[a, a] '[a]
(-.) = op2' $ \x y -> (only_ (x - y), maybe (1, -1) (\g -> (g, -g)) . head')
{-# INLINE (-.) #-}

-- | Optimized version of @'op1' ('*')@.
(*.) :: Num a => Op '[a, a] '[a]
(*.) = op2' $ \x y -> (only_ (x * y), maybe (y, x) (\g -> (y*g, x*g)) . head')
{-# INLINE (*.) #-}

-- | Optimized version of @'op1' ('/')@.
(/.) :: Fractional a => Op '[a, a] '[a]
(/.) = op2' $ \x y -> (only_ (x / y), maybe (1/y, -x/(y*y)) (\g -> (g/y, -g*x/(y*y))) . head')
{-# INLINE (/.) #-}

-- | Optimized version of @'op1' ('**')@.
(**.) :: Floating a => Op '[a, a] '[a]
(**.) = op2' $ \x y -> ( only_ (x ** y)
                       , let dx = y*x**(y-1)
                             dy = x**y*log(x)
                         in  maybe (dx, dy) (\g -> (g*dx, g*dy)) . head'
                       )
{-# INLINE (**.) #-}

-- | Optimized version of @'op1' 'negate'@.
negateOp :: Num a => Op '[a] '[a]
negateOp = op1' $ \x -> (only_ (negate x), maybe (-1) negate . head')
{-# INLINE negateOp  #-}

-- | Optimized version of @'op1' 'signum'@.
signumOp :: Num a => Op '[a] '[a]
signumOp = op1' $ \x -> (only_ (signum x), const 0 . head')
{-# INLINE signumOp  #-}

-- | Optimized version of @'op1' 'abs'@.
absOp :: Num a => Op '[a] '[a]
absOp = op1' $ \x -> (only_ (abs x), maybe (signum x) (* signum x) . head')
{-# INLINE absOp #-}

-- | Optimized version of @'op1' 'recip'@.
recipOp :: Fractional a => Op '[a] '[a]
recipOp = op1' $ \x -> (only_ (recip x), maybe (-1/(x*x)) ((/(x*x)) . negate) . head')
{-# INLINE recipOp #-}

-- | Optimized version of @'op1' 'exp'@.
expOp :: Floating a => Op '[a] '[a]
expOp = op1' $ \x -> (only_ (exp x), maybe (exp x) (exp x *) . head')
{-# INLINE expOp #-}

-- | Optimized version of @'op1' 'log'@.
logOp :: Floating a => Op '[a] '[a]
logOp = op1' $ \x -> (only_ (log x), (/x) . fromMaybe 1 . head')
{-# INLINE logOp #-}

-- | Optimized version of @'op1' 'sqrt'@.
sqrtOp :: Floating a => Op '[a] '[a]
sqrtOp = op1' $ \x -> (only_ (sqrt x), maybe (0.5 * sqrt x) (/ (2 * sqrt x)) . head')
{-# INLINE sqrtOp #-}

-- | Optimized version of @'op2' 'logBase'@.
logBaseOp :: Floating a => Op '[a, a] '[a]
logBaseOp = op2' $ \x y -> ( only_ (logBase x y)
                           , let dx = - logBase x y / (log x * x)
                             in  maybe (dx, 1/(y * log x))
                                       (\g -> (g*dx, g/(y * log x)))
                                   . head'
                           )
{-# INLINE logBaseOp #-}

-- | Optimized version of @'op1' 'sin'@.
sinOp :: Floating a => Op '[a] '[a]
sinOp = op1' $ \x -> (only_ (sin x), maybe (cos x) (* cos x) . head')
{-# INLINE sinOp #-}

-- | Optimized version of @'op1' 'cos'@.
cosOp :: Floating a => Op '[a] '[a]
cosOp = op1' $ \x -> (only_ (cos x), maybe (-sin x) (* (-sin x)) . head')
{-# INLINE cosOp #-}

-- | Optimized version of @'op1' 'tan'@.
tanOp :: Floating a => Op '[a] '[a]
tanOp = op1' $ \x -> (only_ (tan x), (/ cos x^(2::Int)) . fromMaybe 1 . head')
{-# INLINE tanOp #-}

-- | Optimized version of @'op1' 'asin'@.
asinOp :: Floating a => Op '[a] '[a]
asinOp = op1' $ \x -> (only_ (asin x), (/ sqrt(1 - x*x)) . fromMaybe 1 . head')
{-# INLINE asinOp #-}

-- | Optimized version of @'op1' 'acos'@.
acosOp :: Floating a => Op '[a] '[a]
acosOp = op1' $ \x -> (only_ (acos x), (/ sqrt (1 - x*x)) . maybe (-1) negate . head')
{-# INLINE acosOp #-}

-- | Optimized version of @'op1' 'atan'@.
atanOp :: Floating a => Op '[a] '[a]
atanOp = op1' $ \x -> (only_ (atan x), (/ (x*x + 1)) . fromMaybe 1 . head')
{-# INLINE atanOp #-}

-- | Optimized version of @'op1' 'sinh'@.
sinhOp :: Floating a => Op '[a] '[a]
sinhOp = op1' $ \x -> (only_ (sinh x), maybe (cosh x) (* cosh x) . head')
{-# INLINE sinhOp #-}

-- | Optimized version of @'op1' 'cosh'@.
coshOp :: Floating a => Op '[a] '[a]
coshOp = op1' $ \x -> (only_ (cosh x), maybe (sinh x) (* sinh x) . head')
{-# INLINE coshOp #-}

-- | Optimized version of @'op1' 'tanh'@.
tanhOp :: Floating a => Op '[a] '[a]
tanhOp = op1' $ \x -> (only_ (tanh x), (/ cosh x^(2::Int)) . fromMaybe 1 . head')
{-# INLINE tanhOp #-}

-- | Optimized version of @'op1' 'asinh'@.
asinhOp :: Floating a => Op '[a] '[a]
asinhOp = op1' $ \x -> (only_ (asinh x), (/ sqrt (x*x + 1)) . fromMaybe 1 . head')
{-# INLINE asinhOp #-}

-- | Optimized version of @'op1' 'acosh'@.
acoshOp :: Floating a => Op '[a] '[a]
acoshOp = op1' $ \x -> (only_ (acosh x), (/ sqrt (x*x - 1)) . fromMaybe 1 . head')
{-# INLINE acoshOp #-}

-- | Optimized version of @'op1' 'atanh'@.
atanhOp :: Floating a => Op '[a] '[a]
atanhOp = op1' $ \x -> (only_ (atanh x), (/ (1 - x*x)) . fromMaybe 1 . head')
{-# INLINE atanhOp #-}

