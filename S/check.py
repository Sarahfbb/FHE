import concrete.compiler; 
print("GPU enabled: ", concrete.compiler.check_gpu_enabled())
print("GPU available: ", concrete.compiler.check_gpu_available())