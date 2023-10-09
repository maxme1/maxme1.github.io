---
slug: pascal-llvm-3
date: 2023-10-08
categories:
  - LLVM
comments: true
---

# Compiling Pascal with LLVM: Part 3

## Typing

Today we're going to write a bit less code that last time. But don't worry, I'll compensate that with a healthy scoop of
(hopefully) new concepts!

<!-- more -->

### Adding types

First, let's fix all the crutches we left in our parser. And we'll start this by adding types.

```python
from dataclasses import dataclass

hashable = dataclass(unsafe_hash=True, repr=True)


class DataType:
    """ Base class for all data types """


class VoidType(DataType):
    pass


class BooleanType(DataType):
    pass


class CharType(DataType):
    pass


@hashable
class SignedInt(DataType):
    bits: int


@hashable
class Floating(DataType):
    bits: int


@hashable
class Pointer(DataType):
    type: DataType


@hashable
class Reference(DataType):
    type: DataType

@hashable
class StaticArray(DataType):
    dims: tuple[tuple[int, int]]
    type: DataType

@hashable
class DynamicArray(DataType):
    type: DataType

@hashable
class Field:
    name: str
    type: DataType

@hashable
class Record(DataType):
    fields: tuple[Field]

@hashable
class Signature:
    args: tuple[DataType]
    return_type: DataType

@hashable
class Function(DataType):
    signatures: tuple[Signature]
```

So, we're representing types with... Python classes. Each type is an instance of `DataType`. We have some basic stuff
like bool, void, ints and floats with different numbers of bits and two kinds pointers:
[pointers and references](https://stackoverflow.com/questions/57483/what-are-the-differences-between-a-pointer-variable-and-a-reference-variable).
There isn't much difference between them at runtime, it's more about language semantics. Finally, we have more complex
types like static (with dimensions known at compile time) or dynamic arrays and records 
([structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language)), if you're coming from C).
Finally, we introduce a special type for overloaded functions. Each of them is just a collection of `Signature`s - the
types of its arguments, and the return type.

Also let's create some useful types:

```python
Ints = Byte, Integer = SignedInt(8), SignedInt(32)
Floats = Real, = Floating(64),
Void, Boolean, Char = VoidType(), BooleanType(), CharType()
TYPE_NAMES = {
    'integer': Integer,
    'real': Real,
    'char': Char,
    'byte': Byte,
    'boolean': Boolean,
}
```

We'll use `TYPE_NAMES` the parser later.
Now let's fix the parser, and add more types to it.

First we'll fix the nodes, by changing all `type: str` to `type: DataType`.

```python
# I keep all the types in `types.py`
from . import types

@unique
class Const:
    value: Any
    type: types.DataType

@unique
class Definitions:
    names: tuple[Name]
    type: types.DataType

@unique
class ArgDefinition:
    name: Name
    type: types.DataType

@unique
class Function:
    name: Name
    args: tuple[ArgDefinition]
    variables: tuple[Definitions]
    body: tuple[Any]
    return_type: types.DataType
```

#### Constants

```python
def _primary(self):
    match self.peek().type:
        case TokenType.NUMBER:
            body = self.consume().string
            if '.' not in body:
                value = int(body)
                for kind in types.Ints:
                    if value.bit_length() < kind.bits:
                        return Const(value, kind)

            return Const(float(body), types.Real)

        case TokenType.STRING:
            value = self.consume().string
            if not value.startswith("'"):
                raise ParseError('Strings must start and end with apostrophes')
            value = eval(value).encode() + b'\00'
            return Const(value, types.StaticArray(((0, len(value)),), types.Char))

        # ... other cases are unchanged
```

Pretty straightforward. Strings are now just arrays of chars, floats and ints got a type instead of 'integer' and 
'real'. Also, we're being a bit smarter here, and trying to pack integers in the smallest number of bits possible.
So `1` will be of type `Byte`, while `1000` is an `Integer`. Pascal has automatic type upcasting, so this is ok.

#### Definitions

Now this is the most tedious part. We'll have to add support for a bunch of type definitions. 
It's pretty straightforward though, so I won't waste you time describing what's going on here.

```python
def _type(self):
    if self.consumed(TokenType.CIRCUMFLEX):
        return types.Pointer(self._type())

    if self.consumed(TokenType.NAME, string='array'):
        if self.consumed(TokenType.LSQB):
            # true array
            dims = [self._array_dims()]
            while self.consumed(TokenType.COMMA):
                dims.append(self._array_dims())
            self.consume(TokenType.RSQB)

            self.consume(TokenType.NAME, string='of')
            internal = self._type()
            return types.StaticArray(tuple(dims), internal)

        # just a pointer
        self.consume(TokenType.NAME, string='of')
        internal = self._type()
        return types.DynamicArray(internal)

    # string is just a special case of an array
    if self.consumed(TokenType.NAME, string='string'):
        if self.consumed(TokenType.LSQB):
            dims = self._array_dims(),
            self.consume(TokenType.RSQB)
            return types.StaticArray(dims, types.Char)

        return types.DynamicArray(types.Char)

    if self.consumed(TokenType.NAME, string='record'):
        fields = []
        while not self.consumed(TokenType.NAME, string='end'):
            definition = self._definition()
            for name in definition.names:
                fields.append(types.Field(name.name, definition.type))
        return types.Record(tuple(fields))

    kind = self.consume(TokenType.NAME).string.lower()
    return types.TYPE_NAMES[kind]
```

#### Functions

Finally, let's fix that ugly crutch in `_prototype`:

```python
# replace
if mutable:
    kind = f'reference({kind})'

# by
if mutable:
    kind = types.Reference(kind)
```

### The Visitor pattern

With all this in place, after parsing the code we get an 
[abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) or AST. From now on we'll do a lot of tree walking,
which can be quite tedious if you do it without the proper tools. 

In functional languages such a tool is pattern matching. Yes, Python also has 
[pattern matching syntax](https://peps.python.org/pep-0636/), and we even used it already a few times. However, for 
tree walking I feel like it will get messy very quickly, because we'll be forced to cram all the code into a single 
function. 

To keep things nicely separated we'll use the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern).

!!! note ""

    Another book recommendation: the Gang of Four's ["Design Patterns"](https://en.wikipedia.org/wiki/Design_Patterns).

Because we're using Python, a super dynamic language, I'll show you a handy way to implement the visitor pattern without
the need to do type checks or modify the classes we _visit_:

```python
import re
# credit: https://stackoverflow.com/a/1176023
first_cap = re.compile(r'(.)([A-Z][a-z]+)')
all_cap = re.compile(r'([a-z\d])([A-Z])')

def snake_case(name):
    name = first_cap.sub(r'\1_\2', name)
    return all_cap.sub(r'\1_\2', name).lower()

class Visitor:
    def visit(self, node, *args, **kwargs):
        value = getattr(self, f'_{snake_case(type(node).__name__)}')(node, *args, **kwargs)
        value = self.after_visit(node, value, *args, **kwargs)
        return value

    def visit_sequence(self, nodes, *args, **kwargs):
        return tuple(self.visit(node, *args, **kwargs) for node in nodes)

    def after_visit(self, node, value, *args, **kwargs):
        return value
```

All the fun is happening inside `visit`. We're basically doing a kind of dynamic dispatch based on the class name. So,
if we call `Visitor.visit(MyClass())`, inside it will get dispatched to `Visitor._my_class(value)`. 
I'm converting the class name from `CamelCase` to `snake_case` because of [PEP8](https://peps.python.org/pep-0008/).

There's also a useful `after_visit` method, which will come in handy pretty soon. Think of it as a _post-visit hook_.

### Static analysis

Now that we're ready for tree walking, let's see what we actually want to do with our AST.

```pascal linenums="1"
program semantics;
var x, y, z: integer;
    s: string;
    r: array[10] of record
        name: string;
        age: integer;
    end;

function double(x: integer): integer;
var y: integer; // (1)
    z: array[10] of integer; // (2)
begin
    y := 2; // (3) 
    double := x * 2; // (4) 
end;

function double(x: real): real; // (5) 
var y: integer;
    z: array[10] of integer;
begin
    s := 'example'; // (6) 
    double := x * 2;
end;

begin
    r[5].age = 25 + 1; // (7) 
    s := 'start';
    y := 0;
    z := double(y);
    writeln(z);
    writeln; // (8)
end.
```

1.  we can shadow global variables
2.  moreover, we can shadow with a variable of another type
3.  which variable we're referring here? local or global?
4.  this is how we define the return value
5.  we can overload functions
6.  referencing and changing a global variable
7.  assignments can be more complex
8.  we can call functions that take 0 arguments without parentheses

The little program above has all the concepts we need to catch.

**Name tracking**: for each `Name` node, we need to know which variable or function it refers to, which can be 

  - a simple variable
  - _one of_ the overloaded functions that have the same name
  - inside functions, we also have a special variable to assign the return value to

**Type inference**: for each node inside an _expression_ we want to know its type as well as check for type errors and 
do type casting along the way.

**Static dispatch**: because there may be overloaded functions, we need to statically determine which function the user
is referring to.

Now this is a lot of work, and it may be a good idea to split it into several _passes_ of tree walking. But this will
require a bit more code to write, and I want to run my "Hello world" as soon as possible.

So, here's our new visitor

```python
class TypeSystem(Visitor):
    def __init__(self):
        # the actual types of nodes: Node -> DataType
        self.types = {}
        # what the nodes should be cast to: Node -> DataType
        self.casting = {}
        # what each `Name` node is referring to: Node -> Node
        self.references = {}
```

each method for _expression_ nodes will have the following signature:

```python
def _my_node(self, node, expected: DataType | None, lvalue: bool) -> DataType:
    # ...
```

We will specify which type the node is `expected` to have, or `None`, if we don't care. This will come in handy during 
type checks. The methods will return the actual type of the node, e.g. for `1 + 1` it will probably return `Byte`.

The last parameter, `lvalue`, is more interesting. Consider this statement:

```pascal
r[5].age = 25 + 1;
```

As we saw earlier, an assignment is basically two expressions delimited by the `:=` token.

Each expression has a value, and, it turns out that values come in two colors: [`lvalues` and `rvalues`](https://en.wikipedia.org/wiki/Value_(computer_science)). 
`l` and `r`, you guessed it, stand for `left` and `right` respectively. So

```pascal
r[5].age
```

is an `lvalue`, and

```pascal
25 + 1
```

is an `rvalue` in our example.

The only difference between them, at least for us, will be that we'll expect `lvalues` to return a pointer. This makes
sense, because we need an address in memory in which we'll store the `rvalue` we just computed. This requirement also
covers cases like

```pascal
1 + 2 := myfunc(3);
```

The type checker will complain that there's simply no way to compute the pointer to `1 + 2`. 

!!! note ""
    
    Compile-time constants such as `1` or simple expressions like `1 + 2` might be optimized out by the compiler or 
    even stored in a register rather than RAM, and there are no 
    [pointers to registers](https://stackoverflow.com/questions/22154968/pointer-to-register-address)!


Finally, we'll store types and casting information in `self.types` and `self.casting`. The `after_visit` method is the
best place to do this:

```python
def after_visit(self, node, kind, expected=None, lvalue=None):
    # node types
    self.types[node] = kind
    # optionally add type casting, if needed
    if expected is not None:
        if not self.can_cast(kind, expected):
            raise WrongType(kind, expected)
        
        # if no casting needed - just remove it
        if kind == expected:
            self.casting.pop(node, None)
        else:
            self.casting[node] = expected
            kind = expected

    return kind
```

we'll implement `can_cast` later. For now let's just assume it knows all the casting rules, e.g. 
`can_cast(Byte, Integer) is True` but `can_cast(Real, Char) is False`

#### The scope

We need a place to store all the defined variables, a _scope_. Keep in mind, that functions can shadow global variables, 
so when we call a function we enter a new scope. Usually scopes are stored in a stack. When we enter a new scope we:

1. push an empty scope (usually a dict) on top of the stack
2. define new variables by writing to the top scope
3. read variables by traversing the stack: start with the top scope and go down until you find the variable with the 
name you're looking for
4. after you're done just pop the scope from the top of the stack


Let's look at an example:
```pascal
program scope;
var a, b, c: integer;

function f(d: integer): integer:
var c, e: integer;
begin
    f := a + b + c + d + e;
end;

begin
    f(1);
end.
```

After calling the function `f` we enter its scope, and our stack looks like this:

```
    Global            Local
+------------+    +------------+    
| a, b, c, f | -> | c, d, e, f | 
+------------+    +------------+    
```

So, in `f := a + b + c + d + e`, to find `a` and `b` we'll have to traverse the stack, because it's not present in 
the current _Local_ scope.

Note that `f` is present in both scopes, and it even means different things: in _Global_ it's the function `f`, in
_Local_ it's the special variable we're writing the return value to.

!!! note ""
    
    You might ask "what about recursion?" we want to be able to call the function inside its own body. Yes, we'll get 
    to that shortly, don't worry!


Now let's implement all this behaviour. We'll need methods for entering and leaving the scope, as well as defining 
variables and functions and referencing them by name.

```python
from contextlib import contextmanager

class TypeSystem(Visitor):
    def __init__(self):
        self._scopes = []
        self._func_return_names = []
        self.types = {}
        self.casting = {}
        self.references = {}
        self.desugar = {}

    @contextmanager
    def _enter(self):
        self._scopes.append({})
        yield
        self._scopes.pop()
```

So far so good, we're using a `list` here as a stack, and a small 
[context manager](https://realpython.com/python-with-statement/) `_enter` has all the code we need for entering and 
leaving a scope.

Now that we're in a scope, that's how we'll define new names in it:

```python
def _store(self, name: str, kind: types.DataType, payload):
    assert name not in self._scopes[-1]
    self._scopes[-1][name] = kind, payload
```

we store a value of type `kind` by its `name` in the topmost scope. You'll see in a moment what `payload` is for.
The symmetric operation is finding a value by its name:

```python
def _resolve(self, name: str):
    for scope in reversed(self._scopes):
        if name in scope:
            return scope[name]

    raise KeyError(name)
```

The scopes are `reversed`, because we want to iterate from the list's tail - the top of the stack.

And, finally, we store the information that a `Name` refers to a given node like so:

```python
def _bind(self, source, destination):
    self.references[source] = destination
```

The compiler will need this information to quickly find the pointer to the right variable or function.

#### Program

With all the pieces in place, let's start with the main stuff: the program itself and the functions:

```python
def _program(self, node: Program):
    with self._enter():
        # vars
        for definitions in node.variables:
            for name in definitions.names:
                self._store_value(name, definitions.type)

        # functions
        functions = defaultdict(list)
        for func in node.functions:
            functions[func.name.normalized].append(func)
        for name, funcs in functions.items():
            funcs = {f.signature: f for f in funcs}
            self._store(name, types.Function(tuple(funcs)), funcs)

        self.visit_sequence(node.functions)
        self.visit_sequence(node.body)
```


Here `_store_value` is a small util method to store `Name` nodes:

```python
def _store_value(self, name: Name, kind: types.DataType):
    self._store(name.normalized, kind, name)
    self.types[name] = kind
```

This is handy because when we're defining a variable we already know its type, so we can store this info on the spot. 

For functions, though, it's not as simple because of overloading - there are several functions with the
same name. That's why we use a `defaultdict(list)` - this is a simple way to split a set of objects into groups, in our
case - functions. Next, we `_store` a single entry for each function, but save the information that will help us 
differentiate between them in the `payload`.

We're not done with functions yet! Besides storing the functions name in the global scope, we need to `visit` their 
bodies and resolve all the local variables. Note that we start visiting the functions only after we've defined all of
them. This will step makes sure that recursion works as expected, because we can resolve a function's name even if we 
didn't visit its body yet.

Finally, we simply visit each statement in the program's body.

For completeness, here's the body of `visit_sequence`:

```python
def visit_sequence(self, nodes, *args, **kwargs):
    return tuple(self.visit(node, *args, **kwargs) for node in nodes)
```

and `Name.normalized` is just

```python
class Name:
    name: str

    @property
    def normalized(self):
        return self.name.lower()
```

which is handy, because Pascal is case-insensitive.

#### Function

Now that we're done with the hard part, visiting a `Function` node should look almost identical:

```python
def _function(self, node: Function):
    with self._enter():
        self._func_return_names.append((node.name, node.return_type))
        self.types[node.name] = node.return_type

        for arg in node.args:
            self._store_value(arg.name, arg.type)

        for definitions in node.variables:
            for name in definitions.names:
                self._store_value(name, definitions.type)

        self.visit_sequence(node.body)
        self._func_return_names.pop()
```

Most of the code here is about handling this weird "assign to function's name to define the return value" behaviour.
Honestly, I don't know how to handle this better, so here we go: we keep a stack of `(return_type, function_name)`
pairs, which we'll use later to resolve `lvalue`s. At the very end we simply pop this pair from the stack.

The rest is pretty straightforward. Define the variables, don't forget about function arguments (which are also a
kind of local variables) then visit each statement in the body.

#### Assignment

This is one of our main nodes. Assignments are the bridge between `lvalues` and `rvalues`:

```python
def _assignment(self, node: Assignment):
    kind = self.visit(node.target, expected=None, lvalue=True)
    # no need to cast to reference in this case
    if isinstance(kind, types.Reference):
        kind = kind.type

    self.visit(node.value, expected=kind, lvalue=False)
```

First, we get the type of the left side. Here `expected` is `None` because don't care which type we're going to store
in, we only care `what` we'll store there. That's why we visit the right side by passing the type constraint that we
received from the left side.

Additionally, we unwrap the potential `Reference` here: writing to a reference is the same as writing to a regular 
variable, at least from type system's perspective.

#### Const

What's the type of a `Const` node? Simple! We already stored the type while parsing:

```python
def _const(self, node: Const, expected: types.DataType, lvalue: bool):
    if lvalue:
        raise WrongType(node)
    return node.type
```

additionally we make sure here that we're not trying to assign anything to this node, i.e. it's an `rvalue`.

#### Name

Here comes the moment of truth, this little method handles all the references to variables and functions

```python
def _name(self, node: Name, expected: types.DataType, lvalue: bool):
    # assignment to the function's name inside a function is definition of a return value
    if lvalue and self._func_return_names:
        kind, target = self._func_return_names[-1]
        if kind != types.Void and target.name == node.name:
            self._bind(node, target)
            return kind

    kind, target = self._resolve(node.normalized)
    if isinstance(kind, types.Function):
        self.desugar[node] = new = Call(node, ())
        return self._call(new, expected, lvalue)

    self._bind(node, target)
    return kind
```

First we handle our ugly "assign to function name" case. We do this only if

 - it's an `lvalue`
 - we're inside a function i.e. `_func_return_names` isn't empty
 - we're inside a non-Void function (not a procedure), so the return type isn't `Void`
 - the name we're referring to is the same as the function's name

If all these conditions are met, we `bind` the current node to the function's return value.

Otherwise, we `resolve` the name and just `bind` it to the variable we found.

Finally, there's one more case we need to handle. As we saw before, you can call functions with 0 arguments without
parentheses. This is 100% legal:

```pascal
program legal;
begin
    writeln;
end.
```

Looks like in the 70s programmers liked [syntactic sugar](https://en.wikipedia.org/wiki/Syntactic_sugar) 
even more than we do today.

That's why we do another check - if it's a function then it's actually a function call, and we need to replace the
current node with a `Call(node, ())` - we _desugar_ it and store this info to help the compiler.

#### Call

```python
def _call(self, node: Call, expected: types.DataType, lvalue: bool):
    if not isinstance(node.target, Name):
        raise WrongType(node)

    # get all the functions with this name
    kind, targets = self._resolve(node.target.normalized)
    if not isinstance(kind, types.Function):
        raise WrongType(kind)

    # choose the right function
    signature = self._dispatch(node.args, kind.signatures, expected)
    self._bind(node.target, targets[signature])
    return signature.return_type
```

Handling calls is pretty straightforward: 

1. take the `target`, which _must_ be a `Name`, functions aren't 
[first class citizens](https://en.wikipedia.org/wiki/First-class_citizen) in Pascal!
2. `resolve` the name and make sure we've found a function
3. if it's an overloaded function, choose the right variant based on the signatures (static dispatch)
4. `bind` the `Name` node to the function we just chose

All the heavy lifting is done in our `_dispatch` function:

```python
def _dispatch(self, args: Sequence, signatures: Sequence[types.Signature], expected: types.DataType):
    for signature in signatures:
        if len(signature.args) != len(args):
            continue
        if not self.can_cast(signature.return_type, expected):
            continue

        try:
            for arg, kind in zip(args, signature.args, strict=True):
                if isinstance(kind, types.Reference) and not isinstance(arg, Name):
                    raise WrongType('Only variables can be mutable arguments')

                self.visit(arg, expected=kind, lvalue=False)

        except WrongType:
            continue

        return signature

    raise WrongType(args, expected, signatures)
```

Also pretty simple, just loop over all the signatures we have and try to find a match based on the number of `args`,
their types, and the `expected` return type of the function.

In the end we just fail with a `WrongType` if nothing was found.

#### Dereference

We're done with the hard part! The rest should be a piece of cake:

```python
def _dereference(self, node: Dereference, expected: types.DataType, lvalue: bool):
    target = self.visit(node.target, types.Pointer(expected), lvalue)
    return target.type
```

visit the `target` while expecting a pointer, then return the type we point to.

#### GetField

More or less the same here:

```python
def _get_field(self, node: GetField, expected: types.DataType, lvalue: bool):
    target = self.visit(node.target, expected=None, lvalue=False)
    if isinstance(target, types.Reference):
        target = target.type
    if not isinstance(target, types.Record):
        raise WrongType(target)

    for field in target.fields:
        if field.name == node.name:
            return field.type

    raise WrongType(target, node.name)
```

Visit the target, make sure we've got a record, find the right field by name and return its type.

#### GetItem

And here as well:

```python
def _get_item(self, node: GetItem, expected: types.DataType, lvalue: bool):
    target = self.visit(node.target, expected=None, lvalue=True)
    if isinstance(target, types.Reference):
        target = target.type
    if not isinstance(target, (types.StaticArray, types.DynamicArray)):
        raise WrongType(target)

    ndims = len(target.dims) if isinstance(target, types.StaticArray) else 1
    if len(node.args) != ndims:
        raise WrongType(target, node.args)

    args = self.visit_sequence(node.args, expected=types.Integer, lvalue=False)
    args = [x.type if isinstance(x, types.Reference) else x for x in args]
    if not all(isinstance(x, types.SignedInt) for x in args):
        raise WrongType(node)

    return target.type
```

The only difference is that arrays can have multiple indices and we must check that each index is an integer.

#### Unary

Pascal doesn't have many unary operators:

```python
def _unary(self, node: Unary, expected: types.DataType, lvalue: bool):
    if node.op == '@':
        if not isinstance(expected, types.Pointer) or lvalue:
            raise WrongType(node)
        return types.Pointer(self.visit(node.value, expected=expected.type, lvalue=lvalue))

    return self.visit(node.value, expected, lvalue)
```

In case of taking an address (`@`) we check that it's not an `lvalue` and that we're expected to return a `Pointer`.
The rest are just `+`, `-` and `not`, which all return the same type as their argument, so we just visit the `value` 
with the same arguments.

#### Binary

Binary operators, as always, are a bit more interesting. There's a lot of type casting going on with them, e.g. we
want to easily add a `Real` to an `Integer`, which makes perfect sense in most situations.

For me the simplest solution is to treat binary operators as simple functions with 2 arguments. We'll create a 
collection of such functions and use our `_dispatch` method to do all the work:

```python
_numeric = [*types.Ints, *types.Floats]
_homogeneous = {
    '+': _numeric,
    '*': _numeric,
    '-': _numeric,
    '/': _numeric,
    'and': [types.Boolean],
    'or': [types.Boolean],
}
_boolean = {
    '=': _numeric,
    '<': _numeric,
    '<=': _numeric,
    '>': _numeric,
    '>=': _numeric,
    '<>': _numeric,
}
BINARY_SIGNATURES = {
    k: [types.Signature((v, v), v) for v in vs]
    for k, vs in _homogeneous.items()
}
BINARY_SIGNATURES.update({
    k: [types.Signature((v, v), types.Boolean) for v in vs]
    for k, vs in _boolean.items()
})
```

I'm writing from memory here, so I might be wrong, but I'm pretty sure all the operators either return the same type
they received, or a `Boolean` in case of logical operators. That's what the code from above does: it synthetically 
generates a number of valid signatures for binary operators. So the `_binary` method itself becomes as easy as:

```python
def _binary(self, node: Binary, expected: types.DataType, lvalue: bool):
    return self._dispatch([node.left, node.right], BINARY_SIGNATURES[node.op], expected).return_type
```

Not bad at all :sunglasses:

#### Expression statement

We're done with expressions! Now to statements. 

```python
def _expression_statement(self, node: ExpressionStatement):
    self.visit(node.value, expected=None, lvalue=False)
```

Super simple, just visit the expression, don't even care what's the return type.

#### If

```python
def _if(self, node: If):
    self.visit(node.condition, expected=types.Boolean, lvalue=False)
    self.visit_sequence(node.then_)
    self.visit_sequence(node.else_)
```

We visit the condition making sure it's `Boolean`.
Then we unconditionally visit both branches. This contrasts with how `If` is evaluated _at runtime_. For now we're
only interested in variables resolution and expression types, so we _must_ visit both branches.

#### While

```python
def _while(self, node: While):
    self.visit(node.condition, expected=types.Boolean, lvalue=False)
    self.visit_sequence(node.body)
```

Almost same thing here.

#### For

And the final node:

```python
def _for(self, node: For):
    counter = self.visit(node.name, expected=None, lvalue=True)
    if not isinstance(counter, types.SignedInt):
        raise WrongType(counter)

    self.visit(node.start, expected=counter, lvalue=False)
    self.visit(node.stop, expected=counter, lvalue=False)
    self.visit_sequence(node.body)
```

`For` has a counter variable which we assign values to, and it must be an integer.
In rest, this is just a combination of `If` and `While`, nothing new.

### Casting rules

The last piece of the puzzle is the `can_cast` method, that handles all type casting:

```python
def can_cast(self, kind: types.DataType, to: types.DataType) -> bool:
    # we either don't care (to is None) or they're both the same type
    if to is None or kind == to:
        return True

    match kind, to:
        # references are just a wrapper, so we'll ignore them
        case types.Reference(src), _:
            return self.can_cast(src, to)
        case _, types.Reference(dst):
            return self.can_cast(kind, dst)
        
        # static arrays can be viewed as dynamic in some cases, if they're 1-dimensional
        case types.StaticArray(dims, src), types.DynamicArray(dst):
            return len(dims) == 1 and src == dst
        
        # ints can be cast to floats
        case types.SignedInt(_), types.Floating(_):
            return True
    
    # basic upcasting, e.g. Byte -> Integer
    for family in types.SignedInt, types.Floating:
        if isinstance(kind, family) and isinstance(to, family):
            return kind.bits <= to.bits
    
    # no luck
    return False
```

I added comments to the relevant parts, so this should be pretty straightforward.

That's it! Now we have a real-life type system. This took a while, but I hope you found something useful.

You probably noticed that this is already the third post in this series, and still there's no LLVM in sight. In the 
next and final post we'll fix that. Next time we'll use all the concepts we built so far to compile everything to 
[LLVM's IR](https://subscription.packtpub.com/book/programming/9781785280801/1/ch01lvl1sec09/getting-familiar-with-llvm-ir)!
