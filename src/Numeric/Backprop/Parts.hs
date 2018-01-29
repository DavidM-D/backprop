{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE DefaultSignatures      #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE TypeFamilies           #-}

module Numeric.Backprop.Parts (
    Parts(..), Choices(..)
  ) where

import           Data.Type.Product
import           Data.Type.Sum
import           Numeric.Backprop.Iso
import qualified Generics.SOP         as SOP

class Parts bs b | b -> bs where
    parts :: Iso' b (Tuple bs)

    default parts :: (SOP.Generic b, SOP.Code b ~ '[bs]) => Iso' b (Tuple bs)
    parts = gTuple

class Choices bss b | b -> bss where
    choices :: Iso' b (Sum Tuple bss)

    default choices :: (SOP.Generic b, SOP.Code b ~ bss) => Iso' b (Sum Tuple bss)
    choices = gSOP

instance Parts '[a,b]           (a,b)
instance Parts '[a,b,c]         (a,b,c)
instance Parts '[a,b,c,d]       (a,b,c,d)
instance Parts '[a,b,c,d,e]     (a,b,c,d,e)
instance Parts '[a,b,c,d,e,f]   (a,b,c,d,e,f)
instance Parts '[a,b,c,d,e,f,g] (a,b,c,d,e,f,g)

instance Choices '[ '[] , '[a] ] (Maybe a)
instance Choices '[ '[a], '[b] ] (Either a b)
