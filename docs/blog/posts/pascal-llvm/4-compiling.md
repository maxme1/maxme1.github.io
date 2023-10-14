---
slug: pascal-llvm-4
date: 2023-10-13
categories:
  - LLVM
comments: true
---

# Compiling Pascal with LLVM: Part 4

## Compilation

It took several hundreds of lines of code, but we're finally there! Today our compiler will come to life and finally
compute something for us.

<!-- more -->

### A working prototype

Let's start small and build a compiler that works with something super dumb:

```pascal
program add;
begin
    1 + 1;
end.
```

We'll start with that, and work our way up. Here's our compiler class:

```python
from llvmlite import ir


class Compiler(Visitor):
    def __init__(self, ts: TypeSystem):
        self.module = ir.Module()
        self._builders = []

        self._ts = ts
        self._references = ts.references
        # for allocated variables
        self._allocas = {}
        # for functions and strings deduplication
        self._function_names = {}
        self._string_idx = 0
```

As always, let's `visit` the program first:

```python
def _program(self, node: Program):
    main = ir.Function(self.module, ir.FunctionType(ir.VoidType(), ()), '.main')

    with self._enter(main):
        self.visit_sequence(node.body)
        self.builder.ret_void()
```

Here `_enter` is more ore less the same as in our type system:

```python
@contextmanager
def _enter(self, func):
    self._builders.append(ir.IRBuilder(func.append_basic_block()))
    yield
    self._builders.pop()
```

and `self.builder` is just a property:

```python
@property
def builder(self) -> ir.IRBuilder:
    return self._builders[-1]
```

In LLVM each function wants a separate IRBuilder, think of it as a storage for all the statements in the function's
body.

Even though there isn't a `main` function in Pascal as in `C`, LLVM can't just run stuff "in the global scope", so we
have to create here a `.main` function. I chose this name, because it's not a valid function name in Pascal, so we
won't have name collisions.

For now this function is pretty simple. It has no arguments, neither variables, it just goes through the statements
and return `Void` at the end.

#### Expression statement

As I quickly mentioned in the previous post, we'll expect from `lvalues` to return a pointer. This will be the
signature of methods that deal with values:

```python
def _my_node(self, node, lvalue: bool):
    # ...
```

Now we don't need the "`expected`" argument, because all the information regarding types is already present.

So, for expression statements the code is fairly simple:

```python
def _expression_statement(self, node: ExpressionStatement):
    self.visit(node.value, lvalue=False)
```

#### Const

Let's keep digging through the simple stuff first. Here's how we evaluate integer and floating constants:

```python
def _const(self, node: Const, lvalue: bool):
    value = node.value
    match node.type:
        case types.SignedInt(_) | types.Floating(_) as kind:
            return ir.Constant(resolve(kind), value)

        # ... more cases later

    raise ValueError(value)
```

Here `resolve` is the bridge between our own type system and the one from LLVM:

```python
def resolve(kind):
    match kind:
        case types.Void:
            return ir.VoidType()
        case types.Char:
            return ir.IntType(8)
        case types.SignedInt(bits):
            return ir.IntType(bits)
        case types.Floating(64):
            return ir.DoubleType()
        case types.Reference(kind) | types.Pointer(kind) | types.DynamicArray(kind):
            return ir.PointerType(resolve(kind))
        case types.StaticArray(dims, kind):
            size = reduce(mul, [b - a for a, b in dims], 1)
            return ir.ArrayType(resolve(kind), size)
        case types.Record(fields):
            return ir.LiteralStructType([resolve(field.type) for field in fields])

    raise ValueError(kind)
```

As we can see, LLVM's type system is much simpler, e.g. arrays are only 1-dimensional and start always at 0 also
there are no references or dynamic arrays, just pointers.

#### Binary

Once again, binary operators are a headache:

```python
def _binary(self, node: Binary, lvalue: bool):
    left = self.visit(node.left, lvalue)
    right = self.visit(node.right, lvalue)
    kind = self._type(node.left)
    right_kind = self._type(node.right)
    assert kind == right_kind, (kind, right_kind)

    match kind:
        case types.SignedInt(_):
            if node.op in COMPARISON:
                return self.builder.icmp_signed(COMPARISON[node.op], left, right)
            return {
                '+': self.builder.add,
                '-': self.builder.sub,
                '*': self.builder.mul,
                '/': self.builder.sdiv,
            }[node.op](left, right)

        case types.Floating(_):
            if node.op in COMPARISON:
                return self.builder.fcmp_ordered(COMPARISON[node.op], left, right)
            return {
                '+': self.builder.fadd,
                '-': self.builder.fsub,
                '*': self.builder.fmul,
                '/': self.builder.fdiv,
            }[node.op](left, right)

        case types.Boolean:
            return {
                'and': self.builder.and_,
                'or': self.builder.or_,
            }[node.op](left, right)

        case x:
            raise TypeError(x)
```

We visit the left and right operand, and make sure they have the same type, which must be true after type casting.
Here's how we get the type of a node:

```python
def _type(self, node):
    if node in self._ts.casting:
        return self._ts.casting[node]
    return self._ts.types[node]
```

!!! note ""

    Quick reminder: self._ts is an instance of the TypeSystem class, which already analyzed our program.

Then, based on the type we choose from the
[multitude](https://llvmlite.readthedocs.io/en/latest/user-guide/ir/ir-builder.html#arithmetic) of LLVM's operators.
Logical operators get special treatment, because of how they are represented in Pascal:

```python
COMPARISON = {
    '<': '<',
    '<=': '<=',
    '>': '>',
    '>=': '>=',
    '=': '==',
    '<>': '!=',
}
```

We need this mapping to simplify the conversion from `<>` to the modern globally accepted `!=`.

#### Running the code

Now we can finally run our stupid program! We'll need to set a few things up first:

```python
import ctypes

import llvmlite.binding as llvm

from pascal_llvm.compiler import Compiler
from pascal_llvm.parser import Parser
from pascal_llvm.tokenizer import tokenize
from pascal_llvm.type_system import TypeSystem

source = '''
program add;
begin
    1 + 1;
end.
'''
# scan and parse
tokens = tokenize(source)
parser = Parser(tokens)
program = parser._program()
# add types
ts = TypeSystem()
ts.visit(program)
# compile
compiler = Compiler(ts)
compiler.visit(program)
module = compiler.module
# translate
module = llvm.parse_assembly(str(module))
module.verify()
# init
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target = llvm.Target.from_default_triple()
machine = target.create_target_machine()
engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), machine)
# load the code
engine.add_module(module)
engine.finalize_object()
engine.run_static_constructors()
# get the ".main" function pointer
main = ctypes.CFUNCTYPE(None)(engine.get_function_address('.main'))
# call it
main()
```

That's a lot of [glue code](https://en.wikipedia.org/wiki/Glue_code). But it's reusable! From now on you can simply
change the `source` and play around with the compiler as it gets smarter.

At this point this will do basically nothing, but we could print
[LLVM's IR](https://subscription.packtpub.com/book/programming/9781785280801/1/ch01lvl1sec09/getting-familiar-with-llvm-ir)
and see what it thinks of our code. Just call `print(module)`:

```
; ModuleID = ""
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @".main"()
{
.2:
  %".3" = add i8 1, 1
  ret void
}
```

Even if you're not familiar with the syntax, it's pretty clear what's happening here - a function definition, which
adds two i8 (char) constants, stores them in a temporary variable and returns nothing. Just as we intended!

!!! note ""

    If you expected LLVM to optimize out this useless `add` operation, you're totally right. We'll get to that later.

### Variables

Our next milestone is adding variables and tons of code that works with them. First of all, let's define them:

```python
def _program(self, node: Program):
    for definitions in node.variables:
        for name in definitions.names:
            var = ir.GlobalVariable(self.module, resolve(definitions.type), name=name.normalized)
            var.linkage = 'private'
            self._allocas[name] = var

    main = ir.Function(self.module, ir.FunctionType(ir.VoidType(), ()), '.main')
    with self._enter(main):
        self.visit_sequence(node.body)
        self.builder.ret_void()
```

Nothing special, we create a global variable of a given type and store it in our `_allocas` dict. I'm not very familiar
with use-cases for different [linkage types](https://www.llvm.org/docs/LangRef.html#linkage-types), but here "private"
is the way to go - this is basically a global variable _private_ to this module.

#### Name

Now that we can define variables, let's learn to get their values and pointers.

Remember how in the previous post we resolved each `Name` node, and found out what it references? It's time to use this
info:

```python
def _name(self, node: Name, lvalue: bool):
    target = self._references[node]
    ptr = self._allocas[target]
    if lvalue:
        if isinstance(self._type(target), types.Reference):
            ptr = self.builder.load(ptr)
        return ptr
    return self.builder.load(ptr)
```

We get the right pointer first, then dereference it if it's an `rvalue`, otherwise - just return the pointer. A special
case are references, which are represented as pointers: when we write

```pascal
procedure f(var x: integer);
begin
    x := 1;
end;
```

we _pretend_ x is an integer, while under the hood it's a pointer. So, in `_allocas` we'll store  
_a pointer to a pointer_, that's why we need an additional dereferencing step.

#### Dereference

Speaking of dereferencing:

```python
def _dereference(self, node: Dereference, lvalue: bool):
    # always expect a pointer, so lvalue=True
    ptr = self.builder.load(self.visit(node.target, lvalue=True))
    if lvalue:
        return ptr
    return self.builder.load(ptr)
```

More or less the same stuff.

#### Unary

While we're at it, the opposite operation - taking the address - is another unary operator:

```python
def _unary(self, node: Unary, lvalue: bool):
    # getting the address is a special case
    if node.op == '@':
        # just get the name's address
        return self.visit(node.value, lvalue=True)

    value = self.visit(node.value, lvalue)
    match node.op:
        case '-':
            return self.builder.neg(value)
        case 'not':
            return self.builder.not_(value)
        case x:
            raise ValueError(x, node)

```

`@` is, once again, a special case.

#### GetField

LLVM's notion of structs is pretty simple, and is basically indistinguishable from tuples.

```pascal
record
    count: integer;
    percentage: real;
end;
```

This record will have the type `(i32, double)`. This means that we can't access fields by name, and must use integer 
indices instead:

```python
def _get_field(self, node: GetField, lvalue: bool):
    ptr = self.visit(node.target, lvalue=True)
    kind = self._type(node.target)
    if isinstance(kind, types.Reference):
        kind = kind.type
    idx, = [i for i, field in enumerate(kind.fields) if field.name == node.name]
    ptr = self.builder.gep(
        ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), idx)]
    )
    if lvalue:
        return ptr
    return self.builder.load(ptr)
```

We then use the `gep` (get element pointer) command to take the value at the right index in the "tuple".

!!! note ""

    I have no idea why we must pass a list with `ir.Constant(ir.IntType(32), 0)` as the first element, but all the 
    examples I found do it this way. If you can enlighten me, please leave a comment at the bottom of the page :wink:

#### GetItem

As we saw earlier, in the LLVM world there are only 1D arrays that start from 0. That's ok, we just need to figure out
a way to _flatten_ Pascal's arrays. There are many ways to do that, we'll use strides here, more or less in the same
fashion as [numpy](https://numpy.org/) does:

```python
def _get_item(self, node: GetItem, lvalue: bool):
    # we always want a pointer from the parent
    ptr = self.visit(node.target, lvalue=True)
    stride = 1
    dims = self._type(node.target).dims
    idx = ir.Constant(ir.IntType(32), 0)
    for (start, stop), arg in reversed(list(zip(dims, node.args, strict=True))):
        local = self.visit(arg, lvalue=False)
        # upcast to i32
        local = self._cast(local, self._type(arg), types.Integer, False)
        # extract the origin
        local = self.builder.sub(local, ir.Constant(ir.IntType(32), start))
        # multiply by stride
        local = self.builder.mul(local, ir.Constant(ir.IntType(32), stride))
        # add to index
        idx = self.builder.add(idx, local)
        stride *= stop - start

    ptr = self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), idx])
    if lvalue:
        return ptr
    return self.builder.load(ptr)
```

That's a lot of code, but basically this is what it does:

```python
llvm_index = stride1 * (pascal_index1 - index_start1) + stride2 * (pascal_index2 - index_start2) + ...
```

[This StackOverflow question](https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman)
has a great visualization of what's going on.

#### Assignment

The final node that will allow us to _modify_ the variables we just created is the assignment:

```python
def _assignment(self, node: Assignment):
    ptr = self.visit(node.target, lvalue=True)
    value = self.visit(node.value, lvalue=False)
    if isinstance(self._type(node.value), types.Reference):
        value = self.builder.load(value)

    self.builder.store(value, ptr)
```

Also pretty straightforward: get the value from the right, get the address from the left, store one in the other.

### Statements

There aren't a lot of statements in Pascal, so let's dig through them quickly:

#### If

LLVM reasons in blocks. A block is just a sequence of simple operations: no branching, no control flow or jumps.
We just start with the first operation and finish with the last one.

If we _need_ to create a branch, though, we just do it between blocks:

```python
def _if(self, node: If):
    condition = self.visit(node.condition, lvalue=False)
    then_block = self.builder.append_basic_block()
    else_block = self.builder.append_basic_block()
    merged_block = self.builder.append_basic_block()
    self.builder.cbranch(condition, then_block, else_block)

    # then
    self.builder.position_at_end(then_block)
    self.visit_sequence(node.then_)
    self.builder.branch(merged_block)
    # else
    self.builder.position_at_end(else_block)
    self.visit_sequence(node.else_)
    self.builder.branch(merged_block)
    # phi
    self.builder.position_at_end(merged_block)
```

So here's what's going on. We create 3 blocks: two for `if`'s branches and one for the final block. We create a
conditional branch first, this will let us jump to one of the blocks, then we compile _each_ branch. Note that
at the beginning of each branch we enter the corresponding block with `position_at_end`, and at the end we create an
unconditional `branch` (basically a jump) to the final block.

That's more or less how all control flow will be implemented.

#### While

`While` is, in a way, even simpler than `If`:

```python
def _while(self, node: While):
    check_block = self.builder.append_basic_block('check')
    loop_block = self.builder.append_basic_block('for')
    end_block = self.builder.append_basic_block('for-end')
    self.builder.branch(check_block)

    # check
    self.builder.position_at_end(check_block)
    condition = self.visit(node.condition, False)
    self.builder.cbranch(condition, loop_block, end_block)

    # loop
    self.builder.position_at_end(loop_block)
    self.visit_sequence(node.body)
    self.builder.branch(check_block)

    # exit
    self.builder.position_at_end(end_block)
```

At each iteration we check the condition and make a conditional jump. Inside the loop we just compile the body and
unconditionally jump back to the first block. This way we make create a loop. Easy!

#### For

As always, `For` is just a combination of `While` and some increments.

!!! note ""

    It makes you think, maybe we should have desugared this node in the previous post :thinking_face:

```python
def _for(self, node: For):
    name = node.name
    start = self.visit(node.start, lvalue=False)
    stop = self.visit(node.stop, lvalue=False)
    self._assign(name, start)

    check_block = self.builder.append_basic_block('check')
    loop_block = self.builder.append_basic_block('for')
    end_block = self.builder.append_basic_block('for-end')
    self.builder.branch(check_block)

    # check
    self.builder.position_at_end(check_block)
    counter = self.visit(name, lvalue=False)
    condition = self.builder.icmp_signed('<=', counter, stop, 'for-condition')
    self.builder.cbranch(condition, loop_block, end_block)

    # loop
    self.builder.position_at_end(loop_block)
    self.visit_sequence(node.body)
    # update
    increment = self.builder.add(counter, ir.Constant(resolve(self._type(name)), 1), 'increment')
    self._assign(name, increment)
    self.builder.branch(check_block)

    # exit
    self.builder.position_at_end(end_block)
```

I guess only two pieces are interesting here:

```python
start = self.visit(node.start, lvalue=False)
self._assign(name, start)
# and
increment = self.builder.add(counter, ir.Constant(resolve(self._type(name)), 1), 'increment')
self._assign(name, increment)
```

`_assign` is almost identical to `_assignment`:

```python
def _assign(self, name: Name, value):
    target = self._references[name]
    ptr = self._allocas[target]
    if isinstance(self._type(target), types.Reference):
        ptr = self.builder.load(ptr)
    self.builder.store(value, ptr)
```

As we can see, we're writing to the same variable twice, but LLVM is cool with that.

### Functions and calls

We're almost there!

#### Function definitions

We'll define all the functions _after_ defining the global variables but _before_ the main body:

```python
def _program(self, node: Program):
    for definitions in node.variables:
        for name in definitions.names:
            var = ir.GlobalVariable(self.module, resolve(definitions.type), name=name.normalized)
            var.linkage = 'private'
            self._allocas[name] = var
    
    # --- new stuff starts here ---
    for func in node.functions:
        ir.Function(
            self.module, ir.FunctionType(resolve(func.return_type), [resolve(arg.type) for arg in func.args]),
            self._deduplicate(func),
        )
    self.visit_sequence(node.functions)

    main = ir.Function(self.module, ir.FunctionType(ir.VoidType(), ()), '.main')
    with self._enter(main):
        self.visit_sequence(node.body)
        self.builder.ret_void()
```

Just like before, we first define _all_ the functions, and only after that we visit each of their bodies. 
Here `_deduplicate` helps us... uh... deduplicate the function names. Remember that we might have several overloaded 
functions, LLVM can't digest that, so we need to help it a bit:

```python
def _deduplicate(self, node: Function):
    if node not in self._function_names:
        self._function_names[node] = f'function.{len(self._function_names)}.{node.name.name}'
    return self._function_names[node]
```

We create a new unique name for each function. It's a pretty dumb strategy, I admit, but it's dead simple (that's a
good thing!). If I was writing a real compiler, I would add some info regarding argument names and the return value 
instead, this would help a lot with debugging.

#### Function

Now to the function itself. It's more or less the same thing as with the `Program`:

```python
def _function(self, node: Function):
    ret = node.name
    func = self.module.get_global(self._deduplicate(node))
    with self._enter(func):
        if node.return_type != types.Void:
            self._allocate(ret, resolve(node.return_type))

        for arg, param in zip(func.args, node.args, strict=True):
            name = param.name
            arg.name = name.normalized
            self._allocate(name, resolve(param.type), arg)

        for definitions in node.variables:
            for name in definitions.names:
                self._allocate(name, resolve(definitions.type))

        self.visit_sequence(node.body)

        if node.return_type != types.Void:
            self.builder.ret(self.builder.load(self._allocas[ret]))
        else:
            self.builder.ret_void()
```

1. Get the function
2. Enter the scope
3. Define the variables, the arguments and the return value, if any
4. Visit the body
5. Return

The only difference is that we're not defining functions here, because Pascal doesn't support 
[closures](https://en.wikipedia.org/wiki/Closure_(computer_programming)).

We define local variables like so:

```python
def _allocate(self, name: Name, kind: ir.Type, initial=None):
    self._allocas[name] = self.builder.alloca(kind, name=name.normalized)
    if initial is not None:
        self.builder.store(initial, self._allocas[name])
```

`alloca` is LLVM's way to _allocate_ a portion of memory.

#### Call

We can define functions. Time to learn how to call them:

```python
def _call(self, node: Call, lvalue: bool):
    magic = MAGIC_FUNCTIONS.get(node.target.normalized)
    if magic is not None:
        return magic.evaluate(node.args, list(map(self._type, node.args)), self)

    target = self._references[node.target]
    func = self.module.get_global(self._deduplicate(target))
    signature = target.signature

    args = []
    for arg, kind in zip(node.args, signature.args, strict=True):
        if isinstance(kind, types.Reference):
            value = self._allocas[self._references[arg]]
        else:
            value = self.visit(arg, lvalue=False)
        args.append(value)

    return self.builder.call(func, args)
```

If we ignore the _magical_ part, this should be straightforward: 

1. Find the function by its name
2. Compute the arguments
3. Call the function with these arguments

We only have a small hiccup with (2): if it's a mutable argument - we know for sure it's a variable (we checked for that
in the previous post), so we can get its address directly from the `_allocas`.

Now what about those first three lines? They deserve a separate section!

### Even more magic

Previously we learned how to validate the arguments of magic functions. Now let's figure out how to _evaluate_ them.
We'll extend the interface with another method:

```python
class MagicFunction(ABC):
    @classmethod
    @abstractmethod
    def validate(cls, args, visit) -> types.DataType:
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, args, kinds, compiler):
        pass
```

and this is how `WriteLn` would implement it:

```python
class WriteLn(MagicFunction):
    @classmethod
    def validate(cls, args, visit) -> types.DataType:
        for arg in args:
            visit(arg, None, False)
        return types.Void

    @classmethod
    def evaluate(cls, args, kinds, compiler):
        ptr = compiler.string_pointer(format_io(kinds) + b'\n\00')
        return compiler.builder.call(compiler.module.get_global('printf'), [ptr, *compiler.visit_sequence(args, False)])
```

We're basically preparing arguments for [printf](https://en.wikipedia.org/wiki/Printf) - a function from the C world. 
It expects a format spec that depends on argument types:

```python
from jboc import composed

@composed(b' '.join)
def format_io(args):
    for arg in args:
        match arg:
            case types.SignedInt(_):
                yield b'%d'
            case types.Floating(_):
                yield b'%f'
            case types.Char:
                yield b'%c'
            case types.StaticArray(dims, types.Char) if len(dims) == 1:
                yield b'%s'
            case types.DynamicArray(types.Char):
                yield b'%s'
            case kind:
                raise TypeError(kind)
```

We then just find `printf` and call it. But to find it we first need to define it. We'll store all the definitions for
external functions here:

```python
FFI = {
    'printf': ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True),
    'scanf': ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True),
    'getchar': ir.FunctionType(ir.IntType(8), []),
    'rand': ir.FunctionType(ir.IntType(32), []),
    'srand': ir.FunctionType(ir.VoidType(), [ir.IntType(32)]),
    'time': ir.FunctionType(ir.IntType(32), [ir.IntType(32)]),
}
```

!!! note ""

    FFI stands for [foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface)

I defined a bunch of them just in case. Note that `printf` and `scanf` are variadic, just like `writeln` and `readln` 
in Pascal.

Finally, we'll add these definitions in `Compiler`'s constructor:

```python
def __init__(self, ts: TypeSystem):
    # ... other stuff
    
    for name, kind in FFI.items():
        ir.Function(self.module, kind, name)
```

Wait. What's with that `string_pointer` function? We need to pass a pointer to the spec string to `printf`. Here's the 
solution I came up with. We'll make compile-time string global constants like so:

```python
def string_pointer(self, value: bytes):
    kind = ir.ArrayType(ir.IntType(8), len(value))
    global_string = ir.GlobalVariable(self.module, kind, name=f'string.{self._string_idx}.global')
    global_string.global_constant = True
    global_string.initializer = ir.Constant(kind, [ir.Constant(ir.IntType(8), x) for x in value])
    self._string_idx += 1
    return self.builder.gep(
        global_string, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)]
    )
```

We create a global variable with a unique name, `_string_idx` makes sure of that. We initialize it with our string, 
then we take its pointer and return it.

With this code in place, we can also add another branch to `_const`:

```python
def _const(self, node: Const, lvalue: bool):
    value = node.value
    match node.type:
        case types.StaticArray(dims, types.Char) if len(dims) == 1:
            return self.string_pointer(value)
        # ... rest of the cases
```

#### Read

I'll show you a few more implementations real quick:

```python
class Read(MagicFunction):
    @classmethod
    def validate(cls, args, visit) -> types.DataType:
        if not args:
            raise WrongType

        for arg in args:
            visit(arg, None, True)
        return types.Void

    @classmethod
    def evaluate(cls, args, kinds, compiler):
        ptr = compiler.string_pointer(format_io(kinds) + b'\00')
        return compiler.builder.call(compiler.module.get_global('scanf'), [ptr, *compiler.visit_sequence(args, True)])
```

For `Read` we simply call into `scanf` instead of `printf`.

#### ReadLn

Now `ReadLn` is a totally different beast:

```python
class ReadLn(MagicFunction):
    @classmethod
    def validate(cls, args, visit) -> types.DataType:
        for arg in args:
            visit(arg, None, True)
        return types.Void

    @classmethod
    def evaluate(cls, args, kinds, compiler):
        builder = compiler.builder
        ptr = compiler.string_pointer(format_io(kinds) + b'\00')
        builder.call(compiler.module.get_global('scanf'), [ptr, *compiler.visit_sequence(args, True)])
        # ignore the rest of the line: while (getchar() != '\n') {} // ord('\n') == 10
        check_block = builder.append_basic_block('check')
        loop_block = builder.append_basic_block('loop')
        end_block = builder.append_basic_block('end')
        builder.branch(check_block)
        # check
        builder.position_at_end(check_block)
        condition = builder.icmp_signed(
            '!=', builder.call(compiler.module.get_global('getchar'), ()), ir.Constant(ir.IntType(8), 10)
        )
        builder.cbranch(condition, loop_block, end_block)
        # loop
        builder.position_at_end(loop_block)
        builder.branch(check_block)
        # exit
        builder.position_at_end(end_block)
```

We first read several values, then we must skip the rest until we hit the line end.
All the wizardry here is just translating this C code:

```c
scanf(...);
while (getchar() != '\n') {}
```

### Final preparations

Let's deal quickly with the boring stuff.

#### Type casting

Just like before, all type casting is happening in `after_visit`:

```python
def _cast(self, value, src: types.DataType, dst: types.DataType, lvalue: bool):
    # references are just fancy pointers
    if isinstance(src, types.Reference) and not lvalue:
        src = src.type
        value = self.builder.load(value)

    if src == dst:
        return value

    match src, dst:
        case types.SignedInt(_), types.Floating(_):
            return self.builder.sitofp(value, resolve(dst))
        case types.SignedInt(_), types.SignedInt(_):
            return self.builder.sext(value, resolve(dst))
        case types.StaticArray(_), types.DynamicArray(_):
            return value
        # ... more cases here perhaps

    raise NotImplementedError(value, src, dst)


def after_visit(self, node, value, lvalue=None):
    if node in self._ts.casting:
        assert lvalue is not None
        return self._cast(value, self._ts.types[node], self._ts.casting[node], lvalue)
    return value
```

#### Desugaring

To write this one we'll need to extend the `Visitor` interface:

```python
class Visitor:
    def visit(self, node, *args, **kwargs):
        node = self.before_visit(node, *args, **kwargs)
        value = getattr(self, f'_{snake_case(type(node).__name__)}')(node, *args, **kwargs)
        value = self.after_visit(node, value, *args, **kwargs)
        return value

    def before_visit(self, node, *args, **kwargs):
        return node
    
    # ... other methods
```

So, _before_ visiting the node, we can change it and force the compiler to visit something else - perfect for 
desugaring.

And that's how the `Compiler` will implement it:

```python
def before_visit(self, node, *args, **kwargs):
    return self._ts.desugar.get(node, node)
```

!!! note ""

    `d.get(x, x)` is just a quicker way to write `d[x] if x in d else x`

#### Magic registry

Last time we stored all our magic functions in a dict:

```python
MAGIC_FUNCTIONS = {
    'writeln': WriteLn,
}
```

For 2-3 functions this is fine, but it quickly becomes tedious. Here's a cool way to do this automatically:

```python
class MagicFunction(ABC):
    # ... other methods

    def __init_subclass__(cls, **kwargs):
        name = cls.__name__.lower()
        assert name not in MAGIC_FUNCTIONS
        MAGIC_FUNCTIONS[name] = cls
```

`__init_subclass__` is a hook that is triggered right after a new subclass is created, you get the idea.

### Optimization

At this point you should be able to finally print "Hello world!":

```pascal
program main;
begin
    writeln('Hello World!');
end.
```

But there's more we can do! An obvious strong side of LLVM is optimization - it has a lot of different optimizations.
Today we'll just [a few of them](https://llvm.org/docs/Passes.htm) that seem self-explanatory:

```python
# ....
# translate
module = llvm.parse_assembly(str(module))
module.verify()

# optimize
pm_builder = llvm.PassManagerBuilder()
pm = llvm.ModulePassManager()
pm_builder.populate(pm)
# add passes
pm.add_constant_merge_pass()
pm.add_instruction_combining_pass()
pm.add_reassociate_expressions_pass()
pm.add_gvn_pass()
pm.add_cfg_simplification_pass()
pm.add_loop_simplification_pass()
pm.run(module)

# run
llvm.initialize()
# ...
```

The pass manager `pm` will do all the heavy lifting for us:

- merge constants
- reassociate expressions: `(1 + 2) * (2 + 1) -> (1 + 2) * (1 + 2)`
- simplify expressions: `(1 + 2) * (1 + 2) -> (1 + 2) ** 2`
- simplify branching: control flow and loops
- remove redundant instructions

You can print the `module` before and after `pm.run(module)` and see the difference. Pretty cool, right?

### ~

If you're reading this, thanks for sticking around! I hope you enjoyed reading this series of posts as much as I enjoyed
implementing all this stuff. 

There are a ton of ways one could improve this implementation, and it's by no means complete, but I feel like it still
gives a good perspective of what are the key components. 
Let me know in the comments if there are some interesting aspects that I didn't cover!
