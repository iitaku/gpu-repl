project(gpu-repl)

cmake_minimum_required(VERSION 2.8)

function(add_target_compile_flag target flag)
    get_property(current_flags TARGET ${target} PROPERTY COMPILE_FLAGS)
    set_target_properties(${target}
                          PROPERTIES COMPILE_FLAGS
                          "${current_flags} ${flag}")
endfunction(add_target_compile_flag)

function(add_target_link_flag target flag)
    get_property(current_flags TARGET ${target} PROPERTY LINK_FLAGS)
    set_target_properties(${target}
                          PROPERTIES LINK_FLAGS
                          "${current_flags} ${flag}")
endfunction(add_target_link_flag)

add_executable(gpu-repl repl.cc)

execute_process(COMMAND llvm-config --cxxflags OUTPUT_VARIABLE stdout)
string(STRIP "${stdout}" llvm_cxxflags)

execute_process(COMMAND llvm-config --ldflags OUTPUT_VARIABLE stdout)
string(STRIP "${stdout}" llvm_ldflags)

add_target_compile_flag(gpu-repl "${llvm_cxxflags} -fno-rtti -I /usr/local/cuda/include")
add_target_link_flag(gpu-repl "${llvm_ldflags} -L /usr/local/cuda/lib64")

target_link_libraries(gpu-repl
  LLVMNVPTXCodeGen
  LLVMSelectionDAG
  LLVMAsmPrinter
  LLVMMCParser
  LLVMCodeGen
  LLVMScalarOpts
  LLVMInstCombine
  LLVMTransformUtils
  LLVMipa
  LLVMAnalysis
  LLVMNVPTXDesc
  LLVMNVPTXInfo
  LLVMNVPTXAsmPrinter
  LLVMTarget
  LLVMMC
  LLVMObject
  LLVMCore
  LLVMSupport
  cuda
  cudart
  pthread
  dl
  )
