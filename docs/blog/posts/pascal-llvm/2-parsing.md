---
slug: pascal-llvm-2
date: 2023-10-01
categories:
  - LLVM
---

# Compiling Pascal with LLVM: Part 2

## Parsing

Now to the fun part. We'll be using a recursive parser, just like in
[Crafting interpreters](https://craftinginterpreters.com/parsing-expressions.html),
so you can easily skip this part if you're familiar with the technique.
I made a slight modification to the parsing of [binary operators](#binary-operators) though.

<!-- more -->


!!! note ""

    I wrote the parser for Pascal's grammar from memory (a _very_ long-term memory), and apparently ended up with a less
    restrictive grammar. I might fix that in the future though.

We'll start with a simple class with several helper methods:

```python
from tokenize import TokenInfo
from more_itertools import peekable
from ..tokenizer import TokenType


# just for convenience
class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.tokens = peekable(tokens)

    def consume(self, *types: TokenType, string: str | None = None) -> TokenInfo:
        if not self.matches(*types, string=string):
            raise ParseError(self.peek(), types, string)
        return next(self.tokens)

    def consumed(self, *types: TokenType, string: str | None = None) -> bool:
        success = self.matches(*types, string=string)
        if success:
            self.consume()
        return success

    def peek(self) -> TokenInfo:
        if not self.tokens:
            raise ParseError
        return self.tokens.peek()

    def matches(self, *types: TokenType, string: str | None = None) -> bool:
        if not self.tokens:
            return False
        token = self.peek()
        if types and token.type not in types:
            return False
        if string is not None and token.string.lower() != string.lower():
            return False
        return True
```

basically all they do is consume the next token while checking some constraints.

Next, we'll build a collection
of [nodes](https://github.com/maxme1/pascal-llvm/blob/master/pascal_llvm/parser/nodes.py), which the parser will have to
generate.

### Expressions

Expressions are the parts of code that produce a value.
For example, these

```pascal
1 + 1
f(1, 2, 3)
1 + f(2) + x.count * array[0]
```

are expressions, but this isn't

```pascal
if x = 1 then
begin
end;
```

!!! note ""

    Technically, not all calls produce values, because Pascal makes a distinction between functions and procedures,
    but here we'll treat procedures as functions that return **void**, and let the type system worry about them.

#### Literals

The simplest expression is a literal like `1` or `'abc'`.

```python
from dataclasses import dataclass
from typing import Any

# we want each node to be unique, so that two nodes with same fields still won't be equal  
unique = dataclass(eq=False)


@unique
class Const:
    value: Any
    type: str
```

next we'll add a `_primary` method to the `Parser`. Which will return this node:

```python
# TokenType was defined in the previous post
from .tokenizer import TokenType


class Parser:
    # ...
    
    # expression is just a convenience method for now
    def _expression(self):
        return self._primary()

    def _primary(self):
        match self.peek().type:
            case TokenType.NUMBER:
                body = self.consume().string
                if '.' not in body:
                    return Const(int(body), 'integer')
                return Const(float(body), 'real')

            case TokenType.STRING:
                value = self.consume().string
                if not value.startswith("'"):
                    raise ParseError('Strings must start and end with apostrophes')
                value = eval(value).encode() + b'\00'
                return Const(value, 'string')

            case _:
                raise ParseError(self.peek())
```

!!! note ""

    Here we use [null-terminated strings](https://en.wikipedia.org/wiki/Null-terminated_string), which is
    not very Pascal'ish. However, this will simplify things for us when we'll be working with C functions. 

and now you can spin up the whole thing:

```python
text = '1'
parser = Parser(tokenize(text))
print(parser._expression())
```

this should give you something like `Const(value=1, type='integer')`, and if you try and call it with

```python
text = 'x + y'
```

it would raise a `ParseError`. So far so good!

#### Names

While we're at it, let's add another simple node - name access. We'll emit it each time someone references a variable
or function by its name:

```python
@unique
class Name:
    name: str
```

this will be another case in the `_primary` function:

```python
case TokenType.NAME:
    return Name(self.consume().string)
```

#### Tails

Next we'll parse something more interesting:

```pascal
f(1, 2, x, y)
student.name
array[index]
myPointer^
```

So, function calls, field access, array indexing and pointers dereferencing. As usual let's add some nodes first:

```python
@unique
class GetItem:
    target: Any
    args: tuple[Any]


@unique
class GetField:
    target: Any
    name: str


@unique
class Dereference:
    target: Any


@unique
class Call:
    target: Any
    args: tuple[Any]
```

and a new method `_tail`. Let's start with pointers:

```python
class Parser:
    # ...
    
    def _expression(self):
        return self._tail()

    def _tail(self):
        target = self._primary()
        while self.matches(TokenType.LSQB, TokenType.DOT, TokenType.CIRCUMFLEX, TokenType.LPAR):
            match self.consume().type:
                case TokenType.CIRCUMFLEX:
                    target = Dereference(target)

                # ... other cases here

        return target
```

So, what's going on here?

1. First we parse the _target_ - a constant or a variable name. This is exactly what `_primary` returns
2. Next, we check whether it has a _tail_ - a dereferencing operator, a field access etc
3. Finally, we wrap the target in a new node

Also, because we use a while loop here, we'll be able to parse stuff like

```pascal
f(1, 2)[3].names[0]^
```

Now let's add this logic. Field access is also as simple as

```python
case TokenType.DOT:
    name = self.consume(TokenType.NAME).string
    target = GetField(target, name)
```

Here we already consumed the dot in the `match self.consume...` part, so we consume the second token, which must be
a name, and finally we wrap the target in the `GetField` node.

Finally, two more interesting tails:

```python
# array indexing
case TokenType.LSQB:
    args = [self._expression()]
    while self.consumed(TokenType.COMMA):
        args.append(self._expression())
    self.consume(TokenType.RSQB)
    target = GetItem(target, tuple(args))

# function call
case TokenType.LPAR:
    args = []
    while not self.matches(TokenType.RPAR):
        if args:
            self.consume(TokenType.COMMA)
        args.append(self._expression())
    self.consume(TokenType.RPAR)
    target = Call(target, tuple(args))
```

In both cases we parse a sequence of arguments separated by commas, and consume the matching bracket at the end.

#### Unary operators

In case of tails we were able to use a `while` loop, because we _first_ needed to parse the target, and then an
indefinite number of tails. In case of unary operators, though, the target comes at the very end, which means that we
first need to consume all the unary operators, push them into a stack, then consume the target, and wrap it while
unwinding the stack. This sounds like recursion to me:

```python
def _unary(self):
    if self.peek().string.lower() in ('@', 'not', '-', '+'):
        return Unary(self.consume().string.lower(), self._unary())
    return self._tail()
```

and let's not forget the node:

```python
@unique
class Unary:
    op: str
    value: Any
```

#### Binary operators

Binary operators are particularly interesting, because with them the notion of `priority` or `precedence` arises very
naturally. If you're not a [lisper](https://en.wikipedia.org/wiki/Lisp_(programming_language)), then probably you
wouldn't like writing

```pascal
1 + 2 * 3 - 4 / 5 + 6
```

as

```pascal
(((1 + (2 * 3)) - (4 / 5)) + 6)
```

this is where operators precedence kicks in.

!!! note ""

    By the way, you should definitely check out [SICP](https://en.wikipedia.org/wiki/Structure_and_Interpretation_of_Computer_Programs),
    it's a timeless classic!


We'll parse that in the following way. First, let's group the operators by precedence:

```python
PRIORITIES = {
    '*': 1,
    '/': 1,
    'div': 1,
    'mod': 1,
    'and': 1,
    '+': 2,
    '-': 2,
    'or': 2,
    '>': 3,
    '>=': 3,
    '<=': 3,
    '<': 3,
    '=': 4,
    '<>': 4,
    'in': 4,
}
MAX_PRIORITY = max(PRIORITIES.values())
```

next we start from the operations with the lowest precedence (max priority), and try to build its arguments from terms
of higher precedence:

```python
def _binary(self, priority):
    if priority <= 0:
        return self._unary()

    left = self._binary(priority - 1)
    while self.peek().string in PRIORITIES:
        op = self.peek().string
        current = PRIORITIES.get(op)
        # only consume the operation with the same priority
        if current != priority:
            break

        self.consume()
        right = self._binary(current - 1)
        left = Binary(op, left, right)

    return left
```

Basically here we're saying, that

```
1 * 2 + 3 * 4
```

is, first of all, a `+` with its arguments `1 * 2` and `3 * 4`. This is why both `left` and `right` are assigned
to operators with `priority - 1`. At the end we just fallback to `_unary` when we run out of operators.

Finally, as always, the node:

```python
@unique
class Binary:
    op: str
    left: Any
    right: Any
```

and also let's update the `_expression`:

```python
def _expression(self):
    return self._binary(MAX_PRIORITY)
```

#### Parenthesis

Yes, we just got rid of parentheses with the help of precedence. Now let's add them back in!

Parentheses have a very high priority, higher than any binary or unary operation or even a function call. So, `_primary`
is the best place for them:

```python
def _primary(self):
    match self.peek().type:
        case TokenType.LPAR:
            self.consume()
            value = self._expression()
            self.consume(TokenType.RPAR)
            return value

        # ... previous code
```

We don't need a separate node for this branch, because the information about precedence kind of emerges in the syntax 
tree by itself. However, sometimes it's a good idea to keep this information more explicit, e.g. when you're using your
parser to write a code formatter.

This took a while, but we're finally done with parsing expressions. Now you can play with the parser and generate really
complex structures. Note how parentheses influence (sometimes) the structure of the syntax tree. 


### Statements

Until now, we wrote the parser starting with the deepest nodes. This is useful when you want to be able to spin up the 
parser as you progress, and watch how it gets smarter. We'll do the same thing with statements, but in reverse.

#### Program

We'll start with simple programs that don't contain variables or functions:

```python
@unique
class Program:
    body: tuple[Any]
```

Don't worry, we'll update this node later.

```python
def _program(self):
    # program name
    self.consume(TokenType.NAME, string='program')
    self.consume(TokenType.NAME)
    if self.consumed(TokenType.LPAR):
        self.consume(TokenType.NAME)
        self.consume(TokenType.RPAR)

    self.consume(TokenType.SEMI)

    # the body, including the `end`
    statements = self._body()

    # final dot
    self.consume(TokenType.DOT)
    # no more tokens should remain
    if self.tokens:
        raise ParseError(self.tokens.peek())

    return Program(statements)
```

The only interesting part is the name parsing. It turns out, that Pascal allows the programmer to specify the names
of input and output files used, something like 
```pascal
program myName(output_file);
```

I didn't know that until I started googling for larger samples of Pascal code just to test my parser.

#### The body

The body is just a sequence of statements between `begin` and `end`:

```python
from jboc import composed

# ...

@composed(tuple)
def _body(self):
    self.consume(TokenType.NAME, string='begin')
    while not self.matches(TokenType.NAME, string='end'):
        if self.matches(TokenType.NAME, string='begin'):
            yield from self._body()
        else:
            yield self._statement()
        # the semicolon is optional in the last statement
        if not self.matches(TokenType.NAME, string='end'):
            self.consume(TokenType.SEMI)
    self.consume(TokenType.NAME, string='end')
```

Note that there's a special case we're dealing with: `begin`-`end` blocks can be nested. However, unlike most of
modern languages, Pascal doesn't allow variables definition inside blocks, so this nesting doesn't have any semantic 
implications, that's why we're just unpacking it.

!!! note ""
    
    Here `@composed(tuple)` simply gathers all we've `yeild`-ed from the function into a tuple. This is handy in many
    situations.

#### The statement

For now, the only statement we know of is an _expression statement_, like a function call, or a useless arithmetic 
operation that isn't stored anywhere: 

```python
def _statement(self):
    value = self._expression()
    value = ExpressionStatement(value)
    return value
```

with a simple node

```python
@unique
class ExpressionStatement:
    value: Any
```

Aaaaand we're finally ready to parse entire (but simple) programs like this one:

```pascal
program itWorks;
begin
    1 + 2;
    writeln(3, 4, 5);
    writeln(6 + 7, 8 * 9)
end.
```

Pretty cool! The rest is just about adding more content. Let's dig quickly through the basics.

#### If

Because `begin`-`end` is only required for multiple expressions, we'll be using this util function a lot:

```python
def _flexible_body(self):
    if self.matches(TokenType.NAME, string='begin'):
        return self._body()
    return self._statement(),
```

Now let's parse some statements:

```python
def _if(self):
    self.consume(TokenType.NAME, string='if')
    condition = self._expression()
    self.consume(TokenType.NAME, string='then')
    left = self._flexible_body()
    if self.consumed(TokenType.NAME, string='else'):
        right = self._flexible_body()
    else:
        right = ()
    return If(condition, left, right)
```

Yep, basic if-then-else stuff.

```python
@unique
class If:
    condition: Any
    then_: tuple[Any]
    else_: tuple[Any]
```

#### For and while

This is more or less the same

```python
def _for(self):
    self.consume(TokenType.NAME, string='for')
    name = Name(self.consume(TokenType.NAME).string)
    self.consume(TokenType.COLONEQUAL)
    start = self._expression()
    self.consume(TokenType.NAME, string='to')
    stop = self._expression()
    self.consume(TokenType.NAME, string='do')
    body = self._flexible_body()
    return For(name, start, stop, body)

def _while(self):
    self.consume(TokenType.NAME, string='while')
    condition = self._expression()
    self.consume(TokenType.NAME, string='do')
    body = self._flexible_body()
    return While(condition, body)
```

and the nodes 

```python
@unique
class For:
    name: Name
    start: Any
    stop: Any
    body: tuple[Any]

@unique
class While:
    condition: Any
    body: tuple[Any]
```

#### Assignments

With new structures in place, it's time to update the `_statement`. We'll also add assignment syntax while we're at it:

```python
def _statement(self):
    if self.matches(TokenType.NAME, string='if'):
        return self._if()
    if self.matches(TokenType.NAME, string='for'):
        return self._for()
    if self.matches(TokenType.NAME, string='while'):
        return self._while()

    value = self._expression()
    if isinstance(value, (Name, GetItem, GetField, Dereference)) and self.consumed(TokenType.COLONEQUAL):
        value = Assignment(value, self._expression())
    else:
        value = ExpressionStatement(value)
    return value
```

Here what's going on. The last part is all about the difference between

```pascal
student.grade;
```

and

```pascal
student.grade := 10;
```

!!! note ""

    Yes, that's how we assign stuff to variables in Pascal - with the 
    [notorious](https://www.mail-archive.com/python-committers@python.org/msg05628.html) walrus operator 

We can't make a difference beforehand, because the expression can have a super long tail:

```pascal
students[0].math.grades[1] := 10;
```

So we'll just start parsing it like a simple expression, then check the next token. If it's `:=` - we've got an
assignment, otherwise - it's a simple expression statement.

And, of course, assignment deserves its own node:

```python
@unique
class Assignment:
    target: Name | GetItem | GetField | Dereference
    value: Any
```

!!! note ""
    
    Here we only allow assignment for certain expressions, because, say, `1 + 1 := 3` doesn't make a lot of sense


#### Variables

Variable definition in Pascal can only happen in a special block before the body, to complicate things, we are 
allowed to use several such blocks:

```pascal
var x1, x2: integer;
    y: real;
var z: string;
```

so we'll have to do more checks than usual: 

```python
@composed(tuple)
def _variables(self):
    while self.consumed(TokenType.NAME, string='var'):
        while self.peek().string.lower() not in ('var', 'function', 'procedure', 'begin'):
            yield self._definition()
```

Here a definition is a set of names that share a type:

```python
@unique
class Definitions:
    names: tuple[Name]
    type: str
```

and we parse it like so

```python
def _definition(self):
    names = [Name(self.consume(TokenType.NAME).string)]
    while self.consumed(TokenType.COMMA):
        names.append(Name(self.consume(TokenType.NAME).string))
    self.consume(TokenType.COLON)
    kind = self._type()
    self.consume(TokenType.SEMI)
    return Definitions(tuple(names), kind)
```

`_type` _for now_ is just

```python
def _type(self):
    if self.peek().string.lower() not in ('real', 'integer', 'string'):
        raise ParseError(self.peek())
    return self.consume().string.lower()
```

We'll seriously upgrade it in the next post, though.

#### Functions

We're almost there, I promise! Functions are pretty similar to the main program itself: they have the same blocks + 
arguments and a return type (which can be `void`):

```python
@unique
class ArgDefinition:
    name: Name
    type: str

@unique
class Function:
    name: Name
    args: tuple[ArgDefinition]
    variables: tuple[Definitions]
    body: tuple[Any]
    return_type: str
```

!!! note ""

    Why am I using here `Name` instead of `str` as the function and arg names? That's because, unlike you, I know the 
    future! In the next post we'll make heavy use of these names wrapped in `Name`.

All we need to parse a function is just

```python
def _function(self):
    name, args, ret = self._prototype()
    variables = self._variables()
    body = self._body()
    self.consume(TokenType.SEMI)
    return Function(name, args, variables, body, ret)
```

Kinda anticlimactic, right? All the interesting part happens inside the prototype:

```python
def _prototype(self):
    is_func = self.consumed(TokenType.NAME, string='function')
    if not is_func:
        self.consume(TokenType.NAME, string='procedure')

    name = Name(self.consume(TokenType.NAME).string)

    args = []
    if self.consumed(TokenType.LPAR):
        while not self.matches(TokenType.RPAR):
            mutable = self.consumed(TokenType.NAME, string='var')
            group = [Name(self.consume(TokenType.NAME).string)]
            while self.consumed(TokenType.COMMA):
                group.append(Name(self.consume(TokenType.NAME).string))
            self.consume(TokenType.COLON)
            kind = self._type()
            if mutable:
                kind = f'reference({kind})'
            args.extend(ArgDefinition(x, kind) for x in group)
            self.consumed(TokenType.COMMA, TokenType.SEMI)

        self.consume(TokenType.RPAR)

    if is_func:
        self.consume(TokenType.COLON)
        ret = self._type()
    else:
        ret = 'void'
    self.consume(TokenType.SEMI)
    return name, tuple(args), ret
```

Most of this should be pretty easy by now, except maybe the part with `kind = f'reference({kind})'`. What's going on 
here? 

Pascal allows us to define mutable arguments in functions:

```pascal
procedure inc(var x: integer);
begin
    x := x + 1;
end;
```

This function doesn't return anything, it just changes its argument inplace. All this thanks to that little `var`. 
For now I added this ugly crutch `kind = f'reference({kind})'`, in the next post we'll fix it :wink:

#### Final program

With all the parts in place this is what the program will really look like:

```python
@unique
class Program:
    variables: tuple[Definitions]
    functions: tuple[Function]
    body: tuple[Any]
```

and we'll parse it as simply as:

```python
def _program(self):
    self.consume(TokenType.NAME, string='program')
    self.consume(TokenType.NAME)
    if self.consumed(TokenType.LPAR):
        self.consume(TokenType.NAME)
        self.consume(TokenType.RPAR)

    self.consume(TokenType.SEMI)

    variables = self._variables()
    functions = []
    while self.peek().string.lower() in ('function', 'procedure'):
        functions.append(self._function())

    statements = self._body()

    self.consume(TokenType.DOT)
    if self.tokens:
        raise ParseError(self.tokens.peek())

    return Program(variables, tuple(functions), statements)
```

This wasn't as sparse as the previous post, was it? Now, with all the bits in place, let's get to something even cooler:
next time we'll build an entire type system, with compile-time errors and stuff! 
