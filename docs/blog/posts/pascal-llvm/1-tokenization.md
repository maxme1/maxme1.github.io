---
date: 2023-09-29
slug: pascal-llvm-1
categories:
  - LLVM
comments: true
---

# Compiling Pascal with LLVM: Part 1

I always wanted to learn LLVM, but I never felt that there are some useful problems I could solve with it in my line of
work.
Eventually I decided to just have some fun and make something dumb and not useful at all. Yes, we're gonna compile
Pascal! A language that I used for the last time like 15 years ago.

This series of posts is highly inspired by the brilliant
book ["Crafting interpreters"](https://craftinginterpreters.com/) by [Bob Nystrom](https://stuffwithstuff.com/) as well
as the official [tutorial for LLVM](https://llvm.org/docs/tutorial/). If you're into parsers or compilers, you should
definitely check them out!


This is a series of four posts:

1. [Tokenization](/blog/2023/09/29/pascal-llvm-1/)
2. [Parsing](/blog/2023/10/01/pascal-llvm-2/)
3. [Typing](/blog/2023/10/08/pascal-llvm-3/)
4. [Compilation](/blog/2023/10/13/pascal-llvm-4/)

And [here](https://github.com/maxme1/pascal-llvm) you can view the final result!


## Why Pascal?

There are two main reasons:

- Pascal is in a kind of sweet spot: it has a pretty simple grammar, so writing a parser would be fairly easy, but it
  has a lot of constructs not covered in the LLVM tutorial, like references, pointers, record types, functions
  overloading, static typing and so on
- back in school my friend wrote a full-blown [roguelike](https://en.wikipedia.org/wiki/Roguelike) in Pascal, and it
  would be really cool to be able to compile it by myself. So yes, nostalgia plays a role in it, duh.

## What you'll need

Everything is written in `Python3.11` with the [llvmlite](https://github.com/numba/llvmlite) package.
You can find the (almost) full implementation [here](https://github.com/maxme1/pascal-llvm). It lacks some minor stuff,
like [subrange types](https://wiki.freepascal.org/subrange_types), but at this point adding them is more about
implementing a small interface, than inventing something new.

!!! note ""

    Feel free to open an [issue](https://github.com/maxme1/pascal-llvm/issues)
    or [PR](https://github.com/maxme1/pascal-llvm/pulls) if you want to contribute in any way!

<!-- more -->

## Tokenization

The simplest part is tokenization. This is usually the most boring part, at least for me, so I decided to piggyback on
Python's `tokenize` builtin module. We'll just need to fix a few token types, and we're good to go:

```python
import token as _token
import tokenize as tknz
from contextlib import suppress

from more_itertools import peekable

# a nice namespace for token types
TokenType = _token
FIX_EXACT = ';', '(', ')', ',', ':', ':=', '[', ']', '^', '@', '.'


def tokenize(text):
    def generator():
        for x in text.splitlines():
            # `generate_tokens` treats too many blank lines as "end of stream", so we'll patch that
            x = x.strip() or '//empty'
            yield x

    tokens = peekable(tknz.generate_tokens(generator().__next__))
    while tokens:
        token: tknz.TokenInfo = next(tokens)
        # ... patch stuff here
```

Here I'm patching empty lines to avoid some unwanted behavior and I'm wrapping the stream of tokens in `peekable`.
Even though I could simply gather all the tokens into a list beforehand, I like the idea that we'll be needing _at most_
the next token to scan (and later parse) the whole grammar.

Now let's fix the results.

### Exact token types

By default the tokens from `FIX_EXACT` are scanned as `OP`, but we'll need more granular control over them during
parsing:

```python
if token.string in FIX_EXACT:
    token = token._replace(type=_token.EXACT_TOKEN_TYPES[token.string])
```

### Comments

Single-line comments are beginning with `//`

```python
# consume the comment
if token.string == '//':
    start = token.start[0]
    with suppress(StopIteration):
        while tokens.peek().start[0] == start:
            next(tokens)
```

and multi-line comments are marked with `{}`

```python
# and the multiline comment
elif token.string == '{':
    nesting = 1
    
    try:
        while nesting > 0:
            token = next(tokens)
            while token.string != '}':
                if token.string == '{':
                    nesting += 1
                token = next(tokens)
    
            nesting -= 1
    
    except StopIteration:
        raise SyntaxError('Unmatched "{"') from None
```

Here I added support for nested comments. I'm not sure if Pascal originally supported them, but some implementations do,
and it's pretty straightforward anyway.

In both cases we just consume stuff until we get to the comment's end

### Various garbage

Pascal doesn't care about indentation, so:

```python
elif token.type in (
        TokenType.INDENT, TokenType.DEDENT, TokenType.ENCODING, TokenType.ENDMARKER, TokenType.NEWLINE
):  
    # do nothing
    pass
```

### The inequality operator

The strangest part is the inequality operator, which is `<>` in Pascal

```python
# fix the `<>` operator
elif token.string == '<' and tokens and tokens.peek().string == '>':
    # consume the second half
    next(tokens)
    yield token._replace(string='<>')
```

### Ranges

finally, in Pascal you can declare ranges of numbers, such as `1..10`, Python would tokenize it as `1.` (the 1.0 float) 
and `.10` (the 0.1 float). Another patch to the rescue:

```python
# unpack floats
elif token.type == TokenType.NUMBER and (token.string.startswith('.') or token.string.endswith('.')):
    body = token.string
    split = (
        token._replace(string='.', type=TokenType.DOT),
        token._replace(string=body.strip('.')),
    )
    if body.startswith('.'):
        yield from split
    else:
        yield from split[::-1]
```

### The rest

and that's it, other tokens are left unchanged:
```python
else:
    yield token
```

And [here](https://github.com/maxme1/pascal-llvm/blob/master/pascal_llvm/tokenizer.py) you can view the full code for tokenization.

In the next post we'll deal with parsing. That one won't be a 4min read :smiling_imp: 
