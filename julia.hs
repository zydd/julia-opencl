{-# LANGUAGE QuasiQuotes #-}

import Control.Parallel.OpenCL
import Control.Monad.State
import Control.Applicative ((<$>))
import Foreign (castPtr, nullPtr, sizeOf, mallocArray, finalizerFree, new, newArray)
import Foreign.C.Types (CFloat, CUChar, CUInt)
import Foreign.Ptr
import Foreign.ForeignPtr
import Language.C.Quote.OpenCL
import Text.PrettyPrint.Mainland
import Data.Tuple (swap)
import Data.Word
import Codec.Picture

data CLEnv
    = CLEnv
    { clDev :: CLDeviceID
    , clCtx :: CLContext
    , clQue :: CLCommandQueue
    }
        | CLErr

type CL = StateT CLEnv IO

initCLEnv :: Int -> Int -> IO CLEnv
initCLEnv p d = do
    platforms <- clGetPlatformIDs
    let platform = platforms !! p

    devices   <- clGetDeviceIDs platform CL_DEVICE_TYPE_ALL
    let device = devices !! d

    context   <- clCreateContext [CL_CONTEXT_PLATFORM platform] [device] print
    queue     <- clCreateCommandQueue context device []

    return $ CLEnv device context queue

releaseCLEnv :: CLEnv -> IO Bool
releaseCLEnv env = do
    q <- clReleaseCommandQueue (clQue env)
    c <- clReleaseContext (clCtx env)
    return (q && c)

withDevice :: Int -> Int -> CL a -> IO a
withDevice plat dev f = do
    env <- initCLEnv plat dev
    ret <- evalStateT f env
    releaseCLEnv env
    return ret

createKernelFromSource :: String -> String -> CL CLKernel
createKernelFromSource src kern = do
    (CLEnv dev ctx _) <- get
    prog <- liftIO $ clCreateProgramWithSource ctx src
    lift $ clBuildProgram prog [dev] ""
    ret <- lift $ clCreateKernel prog kern
    lift $ clReleaseProgram prog
    return ret

programSource :: String
programSource = pretty 80 $ ppr [cunit|
    __kernel void julia(__global float2 *pc,
                        __global int *w,
                        __global int *h,
                        __global uchar *out)
    {
        int id = get_global_id(0);

        const float2 c = *pc;

        int py = id / *w;
        int px = id - py * *h;

        float2 z;
        z.x = 4.2 * ((float) px / *w - 0.5);
        z.y = -4.2 * ((float) py / *h - 0.5);

        int i;
        for(i = 0; i < 100; i++) {
            float x = z.x * z.x - z.y * z.y + c.x;
            float y = 2.0 * z.y * z.x + c.y;

            if(x * x + y * y > 4.0) break;

            z.x = x;
            z.y = y;
        }

        out[id] = (int) (255.0 * (float) i / 100.0);
    }
    |]

julia :: (CFloat, CFloat) -> Int -> Int -> IO (Image Pixel8)
julia c w h = withDevice 0 0 $ do
    (CLEnv dev ctx que) <- get

    kernel <- createKernelFromSource programSource "julia"

    let len = w * h
        elemSize = sizeOf (0 :: Word8)
        vecSize = len * elemSize

    c_ptr <- lift $ newArray [fst c, snd c]
    c_buf <- lift $ clCreateBuffer ctx [CL_MEM_READ_ONLY,CL_MEM_COPY_HOST_PTR] (2 * sizeOf (0 :: CFloat), castPtr c_ptr)
    lift $ clSetKernelArgSto kernel 0 c_buf

    w_ptr <- lift $ new (fromIntegral w :: CUInt)
    w_buf <- lift $ clCreateBuffer ctx [CL_MEM_READ_ONLY,CL_MEM_COPY_HOST_PTR] (sizeOf (0 :: CUInt), castPtr w_ptr)
    lift $ clSetKernelArgSto kernel 1 w_buf

    h_ptr <- lift $ new (fromIntegral h :: CUInt)
    h_buf <- lift $ clCreateBuffer ctx [CL_MEM_READ_ONLY,CL_MEM_COPY_HOST_PTR] (sizeOf (0 :: CUInt), castPtr h_ptr)
    lift $ clSetKernelArgSto kernel 2 h_buf

    mem_out <- lift $ clCreateBuffer ctx [CL_MEM_WRITE_ONLY] (vecSize, nullPtr)
    lift $ clSetKernelArgSto kernel 3 mem_out

    eventExec <- lift $ clEnqueueNDRangeKernel que kernel [len] [1] []

    output    <- lift $ mallocArray len
    eventRead <- lift $ clEnqueueReadBuffer que mem_out True 0 vecSize (castPtr output) [eventExec]

    liftM (imageFromUnsafePtr w h) (lift (newForeignPtr finalizerFree output))

test = julia (-0.4,0.6) 8192 8192 >>= writePng "out.png"
