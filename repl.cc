
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/IRBuilder.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"

#include <cuda.h>
#include <cuda_runtime.h>

//#include "llvm/LLVMContext.h"
//#include "llvm/DataLayout.h"
//#include "llvm/Module.h"
//#include "llvm/PassManager.h"
//#include "llvm/Pass.h"
//#include "llvm/ADT/Triple.h"
//#include "llvm/IRBuilder.h"
//#include "llvm/Assembly/PrintModulePass.h"
//#include "llvm/Support/IRReader.h"
////#include "llvm/CodeGen/CommandFlags.h"
//#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
//#include "llvm/CodeGen/LinkAllCodegenComponents.h"
//#include "llvm/MC/SubtargetFeature.h"
//#include "llvm/Support/Debug.h"
//#include "llvm/Support/FormattedStream.h"
//#include "llvm/Support/ManagedStatic.h"
////#include "llvm/Support/PluginLoader.h"
//#include "llvm/Support/PrettyStackTrace.h"
//#include "llvm/Support/ToolOutputFile.h"
//#include "llvm/Support/Host.h"
//#include "llvm/Support/Signals.h"
//#include "llvm/Support/TargetRegistry.h"
//#include "llvm/Support/TargetSelect.h"
//#include "llvm/Target/TargetLibraryInfo.h"
//#include "llvm/Target/TargetMachine.h"
#include <stdint.h>

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
using namespace llvm;

#if 0
int main()
{
}
#else
extern "C" {
void LLVMInitializeNVPTXTarget();
void LLVMInitializeNVPTXTargetInfo();
void LLVMInitializeNVPTXTargetMC();
void LLVMInitializeNVPTXAsmPrinter();
} /* extern "C" */

typedef enum {
  kOpAdd,
  kOpSub,
  kOpMul,
  kOpDiv
} Op;

class AstNode;
namespace global {

std::vector<AstNode*> mempool;
IRBuilder<> *builder;

} /* namespace global */

namespace {

std::string strip_edge(const std::string::const_iterator& begin, const std::string::const_iterator& end)
{
  std::string::const_iterator liter;
  for ( liter = begin
      ; liter < end && *liter == ' '
      ; ++liter
      ) { }

  std::string::const_iterator riter;
  for ( riter = end
      ; riter > begin && *riter == ' '
      ; --riter
      ) { }

  return std::string(liter, riter);
}

Op convert_binop(const char op)
{
  switch(op)
  {
    case '+':
      return kOpAdd;
    case '-':
      return kOpSub;
    case '*':
      return kOpMul;
    case '/':
      return kOpDiv;
  }

  std::stringstream ss;
  ss << "invalid op : " << op;

  throw ss.str();
}

char convert_binop(Op op)
{
  switch(op)
  {
    case kOpAdd:
      return '+';
    case kOpSub:
      return '-';
    case kOpMul:
      return '*';
    case kOpDiv:
      return '/';
  }

  throw std::string("invalid op");
}

} /* namespace anonymous */

class AstNode
{
public:
  virtual ~AstNode(void) {}
  
  virtual Value * get(void) = 0;
  virtual std::string str(void) = 0;
};

class BinOpNode : public AstNode
{
public:
  static AstNode * Create(Op op, AstNode *lhs, AstNode *rhs)
  {
    AstNode *node = new BinOpNode(op, lhs, rhs);
    global::mempool.push_back(node);
    return node;
  }

  virtual ~BinOpNode(void) {}

  virtual Value * get(void)
  {
    Value *lvalue = lhs_->get();
    Value *rvalue = rhs_->get();

    switch(op_)
    {
      case kOpAdd:
        return global::builder->CreateAdd(lvalue, rvalue);
      case kOpSub:
        return global::builder->CreateSub(lvalue, rvalue);
      case kOpMul:
        return global::builder->CreateMul(lvalue, rvalue);
      case kOpDiv:
        return global::builder->CreateSDiv(lvalue, rvalue);
    }

    throw std::string("unreachable");
  }

  virtual std::string str(void)
  {
    std::stringstream ss;
    ss << '(' << convert_binop(op_) << ' ' << lhs_->str() << ' ' << rhs_->str() << ')';
    return ss.str();
  }

private:
  BinOpNode(Op op, AstNode *lhs, AstNode *rhs)
    : op_(op), lhs_(lhs), rhs_(rhs) { }

  Op op_;
  AstNode *lhs_;
  AstNode *rhs_;
};

class ValueNode : public AstNode
{
public:
  static AstNode * Create(int32_t value)
  {
    AstNode *node = new ValueNode(value);
    global::mempool.push_back(node);
    return node;
  }

  virtual ~ValueNode(void) {}

  virtual Value * get(void)
  {
    return global::builder->getInt32(value_);
  }

  virtual std::string str(void)
  {
    std::stringstream ss;
    ss << '(' << value_ << ')';
    return ss.str();
  }

private:
  ValueNode(int32_t value) : value_(value) {};

  int32_t value_;
};

void release_mempool(void)
{
  std::vector<AstNode*>::iterator iter;
  for ( iter = global::mempool.begin()
      ; iter < global::mempool.end()
      ; ++iter
      )
  {
    delete (*iter);
  }

  global::mempool.clear();
}

AstNode * parse(std::string buf)
{
  if (buf == "")
  {
    throw std::string("invalid syntax");
  }

  std::string stripped_buf = strip_edge(buf.begin(), buf.end());

  if (*(stripped_buf.begin()) == '(' && *(stripped_buf.end()-1) == ')')
  {
    std::string expr = strip_edge(stripped_buf.begin()+1, stripped_buf.end()-1);

    const char op = *(expr.begin());
    std::string lhs;
    std::string rhs;

    int level = 0;

    const std::string::iterator begin = expr.begin() + expr.find_first_not_of(' ', 1);
    const std::string::iterator end = expr.end();
    std::string::iterator iter;

    for ( iter = begin
        ; iter < end
        ; ++iter)
    {
      if (*iter == '(')
      {
        ++level;
      }
      else if (*iter == ')')
      {
        --level;
      }
      else if (*iter == ' ' && level == 0)
      {
        lhs = std::string(begin, iter);
        rhs = std::string(iter, end);
      }
    }

    return BinOpNode::Create(convert_binop(op), parse(lhs), parse(rhs));
  }
  else
  {
    char *e;
    long value = strtol(stripped_buf.c_str(), &e, 10);

    if (*e != '\0')
    {
      std::stringstream ss;
      ss << "invalid value : " << *e;
      throw ss.str();
    }

    return ValueNode::Create(static_cast<int32_t>(value));
  }

  return NULL;
}

int main(int argc, char *argv[])
{
  const char *prompt = ">>> ";

  // Initialize LLVM subsystems
  LLVMContext context;

  //InitializeAllTargets();
  //InitializeAllTargetMCs();
  //InitializeAllAsmPrinters();
  //InitializeAllAsmParsers();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeUnreachableBlockElimPass(*Registry);

  /* Frontend */
  Module *module = new Module("repl_module", context);

  std::vector<Type*> arg_types;
  arg_types.push_back(PointerType::get(Type::getInt32Ty(context), 1));

  FunctionType *function_type =
    FunctionType::get( Type::getVoidTy(context)
                           , arg_types
                           , false
                           );
  Function *function = cast<Function>(module->getOrInsertFunction("repl_function", function_type));
  BasicBlock::Create(context, "EntryBlock", function);

  NamedMDNode *annotate = module->getOrInsertNamedMetadata("nvvm.annotations");
  std::vector<Value *> vals;
  vals.push_back(function);
  vals.push_back(MDString::get(context, "kernel"));
  vals.push_back(ConstantInt::get(Type::getInt32Ty(context), 1));
  annotate->addOperand(MDNode::get(context, vals));

  /* Backend */
  Triple triple;
  std::string err;
  const Target *target = TargetRegistry::lookupTarget("nvptx", triple, err);

  TargetOptions options;
  //options.LessPreciseFPMADOption = EnableFPMAD;
  //options.NoFramePointerElim = DisableFPElim;
  //options.NoFramePointerElimNonLeaf = DisableFPElimNonLeaf;
  //options.AllowFPOpFusion = FuseFPOps;
  //options.UnsafeFPMath = EnableUnsafeFPMath;
  //options.NoInfsFPMath = EnableNoInfsFPMath;
  //options.NoNaNsFPMath = EnableNoNaNsFPMath;
  //options.HonorSignDependentRoundingFPMathOption =
  //  EnableHonorSignDependentRoundingFPMath;
  //options.UseSoftFloat = GenerateSoftFloatCalls;
  //if (FloatABIForCalls != FloatABI::Default)
  //  options.FloatABIType = FloatABIForCalls;
  //options.NoZerosInBSS = DontPlaceZerosInBSS;
  //options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
  //options.DisableTailCalls = DisableTailCalls;
  //options.StackAlignmentOverride = OverrideStackAlignment;
  //options.RealignStack = EnableRealignStack;
  //options.TrapFuncName = TrapFuncName;
  //options.PositionIndependentExecutable = EnablePIE;
  //options.EnableSegmentedStacks = SegmentedStacks;
  //options.UseInitArray = UseInitArray;
  //options.SSPBufferSize = SSPBufferSize;

  std::auto_ptr<TargetMachine>
    target_machine(target->createTargetMachine(triple.getTriple(), "sm_20", "", options));

  PassManager pm;
  pm.add(new TargetLibraryInfo(triple));
  pm.add(new TargetTransformInfo(target_machine->getScalarTargetTransformInfo(), target_machine->getVectorTargetTransformInfo()));
  pm.add(new DataLayout(*(target_machine->getDataLayout())));

  target_machine->setAsmVerbosityDefault(true);

  std::string ptxcode;
  raw_string_ostream ros(ptxcode);
  formatted_raw_ostream fos(ros);

  target_machine->addPassesToEmitFile(pm, fos, TargetMachine::CGFT_AssemblyFile);

  /* CUDA */
  assert(cudaSetDevice(0) == cudaSuccess);
    
  int32_t h_dst;
  int32_t *d_dstptr;
  assert(cudaMalloc(&d_dstptr, sizeof(int32_t)) == cudaSuccess);
  
  /* Repl */
  std::cout << prompt;
  std::string buf;
  while (std::getline(std::cin, buf))
  {
    try
    {
      function->getEntryBlock().eraseFromParent();
      BasicBlock *bb = BasicBlock::Create(context, "EntryBlock", function);

      global::builder = new IRBuilder<>(bb);

      AstNode *node = parse(buf);

      //std::cout << node->str() << std::endl;

      Argument *dst = function->arg_begin();
      Value *value = node->get();

      global::builder->CreateStore(value, dst);
      global::builder->CreateRetVoid();

      //module->dump();

      pm.run(*module);

      fos.flush();
      
      CUmodule cu_module;
      CUfunction cu_function;
      assert(cuModuleLoadDataEx(&cu_module, ptxcode.c_str(), 0, 0, 0) == CUDA_SUCCESS);
      assert(cuModuleGetFunction(&cu_function, cu_module, "repl_function") == CUDA_SUCCESS);
      assert(cuFuncSetBlockShape(cu_function, 1, 1, 1) == CUDA_SUCCESS);
      assert(cuParamSetv(cu_function, 0, &d_dstptr, sizeof(d_dstptr)) == CUDA_SUCCESS);
      assert(cuParamSetSize(cu_function, sizeof(d_dstptr)) == CUDA_SUCCESS);
      assert(cuLaunchGrid(cu_function, 1, 1) == CUDA_SUCCESS);
      
      assert(cudaMemcpy(&h_dst, d_dstptr, sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
      
      ptxcode.clear();

      release_mempool();

      delete global::builder;
    }
    catch(const std::string& msg)
    {
      std::cout << msg << std::endl << prompt;
      continue;
    }

    std::cout << prompt;
  }

  cudaFree(d_dstptr);
  
  llvm_shutdown();

  return 0;
}
#endif
