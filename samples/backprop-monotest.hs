{-# LANGUAGE GADTs #-}

import           Numeric.Backprop.Mono

testImplicit :: BPOp s N3 N1 Double Double
testImplicit = implicitly $ \(x :* y :* z :* ØV) ->
    (((x * y) + y) * z) :* ØV

testExplicit :: BPOp s N3 N1 Double Double
testExplicit = withInps $ \(x :* y :* z :* ØV) -> do
    xy   <- op2 (*) ~$ (x   :* y :* ØV)
    xyy  <- op2 (+) ~$ (xy  :* y :* ØV)
    xyyz <- op2 (*)        ~$ (xyy :* z :* ØV)
    return $ xyyz :* ØV

main :: IO ()
main = do
    print $ backprop testImplicit (2 :+ 3 :+ 4 :+ ØV)
    print $ backprop testExplicit (2 :+ 3 :+ 4 :+ ØV)
