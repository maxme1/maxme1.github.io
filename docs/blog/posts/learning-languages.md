---
date: 2023-11-11
comments: true
---

# Learning a new language with LLMs

I _love_ learning new languages, both programming and natural ones. It feels a bit magical when you realise how words
in different languages are related and why they sound the way they do.

There's been a few months since I started learning european portuguese. I really like it so far, especially how it
sounds. Currently, I'm at a point where I can confidently read and understand almost any text, and, more or less,
understand what native speakers are telling me. My main problem now is my vocabulary - when I want to say or write
something it takes forever to recall some words.

It's pretty funny because sometimes I just freeze trying to remember something and the person I'm talking to assumes I
didn't understand anything and quickly switches to english.
Yes, they say that practice is the best way to learn a new language, but how long will it take, if the first steps are
so hard? 

Recently I stumbled across the [TinyStories](https://arxiv.org/abs/2305.07759) paper. It's a great read, check it out!
In brief, they generated ~2.7 million stories using the typical vocabulary of a 3-4 year old. They then trained several
super small language models (I'm talking 4-6 orders of magnitude smaller than [GPT-4](https://arxiv.org/abs/2303.08774)
or [Llama 2](https://arxiv.org/abs/2307.09288)) on these stories and proved that models with around 30M parameters are
not only fluent in this restricted subset of english, but also show signs of basic reasoning regarding the text's
content.

!!! note ""

    Hey, if a model with just 30M parameters can do it, then I can too!

Then it hit me. This is a great starting point to increase my vocabulary. I'll translate random texts from TinyStories
to european portuguese and then fix the mistakes I made. Skipping ahead, this works even better than I expected!
I literally feel how I'm memorising new words and constructs. Of course, this effect will saturate over time, but for
now, this is the fastest I've ever learned a new language!

I ended up writing a tool that simplifies most of the boilerplate. Its frontend is written in
[Angular](https://angular.io/), the backend is [FastAPI](https://fastapi.tiangolo.com/) and, of course, I use
[ChatGPT](https://chat.openai.com/) to fix my translations.

As always, [here](https://github.com/maxme1/tiny-stories) you can find the full code.
And [here](https://maxme1.github.io/tiny-stories/) is a live version which can be used in two modes:

1. Static - just a static page with a small subset of the TinyStories (~3k). It will generate a prompt that you'll have
   to manually paste into the chat with ChatGPT. It's not 100% automatic, but it's free, and I tried to simplify the
   process as much as I could.
2. Automatic - you have access to all ~2.7mil stories and all the checks are made though API calls to ChatGPT, but
   you'll need an API token for that, and most importantly, you'll need to trust me that I won't steal it! Sadly,
   OpenAI's API doesn't support [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS), so I have to route all
   the requests though a self-hosted proxy server.

!!! note ""

    You can use the string `free` as the token, in which case it will be replaced with my personal token server-side.
    It's limited to 1 request per 10 minutes ~~because I don't want to go broke~~, but I hope it should be enough. 
    Enjoy!

<!-- more -->

## The code

!!! warning

    I am by no means a web developer, so the TypeScript code might seem like
    [man-made horrors beyond your comprehension](https://knowyourmeme.com/memes/you-may-live-to-see-man-made-horrors-beyond-your-comprehension).

I won't bore you with all the code, because most of it is pretty generic. There are a couple of things that I'd like to
mention though:

- It's not so simple to highlight parts of the text
   in `textarea`. [Here](https://codersblock.com/blog/highlight-text-inside-a-textarea/) you can find a great blog post
   describing the solution I ended up using. It's a bit hacky, but it gets the job done.
- `textarea` is not auto-resizable by default, it has a fixed height. In my case it feels more natural to make it more
   responsive. Here how it's done:
   ```typescript
   import { Directive, ElementRef, HostListener, Renderer2 } from '@angular/core';
   
   @Directive({
     selector: 'textarea[autoResize]'
   })
   export class AutoResizeTextareaDirective {
     constructor(private el: ElementRef<HTMLTextAreaElement>) { }
   
     private adjustTextareaHeight(textarea: HTMLTextAreaElement): void {
       textarea.style.overflow = 'hidden';
       textarea.style.height = 'auto';
       textarea.style.height = textarea.scrollHeight + 1 + 'px';
     }
   
     @HostListener('input', ['$event.target'])
     onInput(textarea: HTMLTextAreaElement): void {
       this.adjustTextareaHeight(textarea);
     }
   
     ngAfterViewInit(): void {
       this.adjustTextareaHeight(this.el.nativeElement);
     }
   
     ngAfterViewChecked(): void {
       this.adjustTextareaHeight(this.el.nativeElement);
     }
   }
   ```
   
   So we're basically just changing the raw underlying html element on any relevant event, such as "input".

- To find out what parts of the text were changed, I used `diffWords` from [jsdiff](https://github.com/kpdecker/jsdiff)
   with a small adjustment. By default `diffWords` will output changed segments in a "tangled" manner:

   ```
   a: Lorem ipsum dolor sit amet
   b: Lorem ipsem dollar sit amet
   diff: Lorem -ipsum- +ipsem+ -dolor- +dollar+ sit amet
   fixed: Lorem -ipsum dolor- +ipsem dollar+ sit amet
   ```

   To my taste it looks nicer when all the changed parts are joined together, so we get an overall smaller diff.

- Finally, I organized the TinyStories dataset into ~1300 chunks, ordered by text length. In total, the text takes
   about 2GiB of space. Theoretically, I _could_ load it all in memory, but I've set the request limit to 1 per
   second anyway, so I don't care much about speed. At runtime, I just find the right chunk, load it into memory (~2MiB),
   and select a random text from it.

That's it! It took me a few hours to build the whole thing, so there's no big revelation at the end - just a small tool
doing its job. I hope you'll find it useful next time you decide to learn a new language :wink:
