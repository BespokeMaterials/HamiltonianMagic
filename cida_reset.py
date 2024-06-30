import pycuda.driver as cuda
import pycuda.autoinit

# Reset the GPU
cuda.Device(0).reset()