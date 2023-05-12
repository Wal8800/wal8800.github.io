---
layout: post
title:  "Writing plain code"
date:   2023-03-15 17:47:48 +1300
categories: jekyll update
---

Looking to improve the clarity of my technical writing, I read "Oxford Guide to Plain English". The book provides different guidelines to help people to write in Plain English. It uses the definition of Plain English from [Plain Language Association International](https://plainlanguagenetwork.org/plain-language/plain-language-around-the-world/). Plain English is defined as "A communication is in plain language if its wording, structure, and design are so clear that the intended audience can easily find what they need, understand what they find, and use that information."

As I read the book, I find the definition and some of the guidelines to be applicable for programming code. In this blog post, we will explore what does plain code means and try to apply some of the Plain English guidelines.

## What is plain code

A piece of code could be plain code if they are so clear that the developer can easily navigate and understand what the code is doing. However, there is an important difference between programming codes and natural languages like English. For a piece of English text, once written and published, the text will mostly be read only. On the other hand, for a code base, it will be read and modify through out it's life time. As a result, we need to consider what impacts should plain code have on modifying the code.

If plain code allows the developer to easily navigate the code and builds a good understanding then the developer should be able to leverage their understanding to know where and how to make changes. Furthermore, plain code should be relatively easier to make the right changes, given that from our good understanding, we know the potential impact of the changes. This means we can come up with a plan to deal with the known issues, mitigate potential risks, and avoid negative consequences.

Does plain code also means reducing the amount of changes needed to meet the new requirement? I think it's mostly depended on the existing implementation. If the existing implementation is not built to be extended for the new requirement then there will be relatively more changes. Requiring less changes for a new requirement is a result of extendable code and I would argue plain code and extendable code are two independent properties. Plain code aims to improve the clarity and readability by having clear structure and design. Whereas extendable code are looking forward to the potential use cases and creating a design that caters to the current and possible future requirements. It's possible to have code that have obscure design and structure but once you finally understand the code, you realise the existing implementation can extends to fit the new use with minimal changes.

Coming back to the definition of plain code, putting it together, **code is in plain code if it's structure and design are so clear that the developer can easily find what they need, understand what they find and make the changes that they want.**

Based on this definition, let's have a look at applying some of the guidelines to help write in plain code.

## Converting negative to positive

The book recommends using positive phrases over negative phrases and only use negative when it is necessary. For example, "Vote for only one candidate only" instead of "Vote for no more than one candidate". For negative phrase, people often need to perform extra mental work as they need to figure out the positive meaning first. By favouring positive phrases, it reduces the amount of mental work and increases clarity. 

For coding, we can favour positive phrases when naming function and variable if it's suitable to the application context. More importantly, I think it's the idea of reducing mental work that can help developers to understand the code quicker. I often find myself simplifying boolean conditions to reduce the mental work for the reader when reading boolean condition.

For example, if you have a nested boolean condition:


```python
not (fruit.is_sweet and fruit.has_seeds)
```

We can apply [DeMorgan Laws](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) and the condition becomes


```python
not fruit.is_sweet or not fruit.has_seeds
```

The reader no longer needs to evaluate the bracket first, they can evaluate the condition as they read from left to right.

Another scenario that I observed that could use a review is checking if a boolean field is false.

```python
not config.disabled
```

In almost all cases, the condition is the simplest and it's clear to the reader. However, if we need to repeatedly check the same conditions, it makes me wonder if we should invert the boolean variable.

```python
config.enabled
```

This means we don't need to have a logical not operator in every if condition and directly use the field's semantic meaning.

Inverting existing conditions across the code base is usually not worth the effort especially if there are a lot of dependencies or require breaking changes. This is something I would consider when adding a new argument or creating a new function.

## Organizing your material in a reader-centred structure

One of the guidelines for Plain English is to organise your material so the reader can see the important information early and navigate the document easily. The reader should be able to quickly extract out what they want and focus on the key ideas. The book describes different structure depending on the purpose of the writing. For example:

**Top heavy triangle** - Putting the most important points first and then subsequent ones. Useful for requesting actions from the reader.

**Question and Answers** - The Question & Answer blocks help to break down information in chunks for the reader to digest. The question also engages the reader by calling them into action as the question can include personal words. For example, _Where can I get more information?_

**Full formal report** - A report has different sections such as Summary, Introduction, Discussion, Conclusion and Recommendation. Useful for detailed administrative, technical report or consultation paper.

Each of these examples organise information so that their target audience can find and read the relevant text immediately. For coding, what structure can we implement to help the developers to read and understand the code?

### Creating structure in the code

There are a lot of existing frameworks and patterns that provide structure to build various software application. For building web application and API service, we can use frameworks such as [Next.js](https://nextjs.org/), [Spring](https://spring.io/) or [Django](https://www.djangoproject.com/). These frameworks provide a skeleton for the developer to write their application code. In addition, they usually comes with utility components so the developer don't have to write the common logic like authentication and logging from scratch.

Under the hood, these frameworks may be using patterns such as [MVC](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) or [MVVM](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93viewmodel) to separate codes that have different concerns. Then once we get into the application code, depending on the functionality, developers can use patterns such as [Strategy pattern](https://en.wikipedia.org/wiki/Strategy_pattern) to alter application behaviour during runtime or [Observer pattern](https://en.wikipedia.org/wiki/Observer_pattern) to create one to many dependency without tightly coupling the objects.

Using these frameworks and patterns helps the developer to navigate the code, build an understanding and make potential changes. For example, separating the code of different concern with MVC allows the developer to know where to look for the view, model and controller logic. On other hand, using established pattern like the Observer pattern, allows the developer to quickly understand how dependency between objects are managed and make changes to it if they need to.

Just like the example structures for English, these frameworks and patterns are aim to solve a specific problem. Are there generic structures that we can apply to our code to improve the clarity and readability? 

From my experience, there are some idea that I would use to structure code within a function to improve clarity and readability. These ideas are not specific to a particular solution or functionality. They are lower level ideas of arranging logic within a function. I found them particular useful when the function is long and contains various different logic. In the next few sections, we will explore these ideas.


**Organise the lines in the function into logical blocks**

We can rearrange lines in the function by:

- Using whitespace for separating logic.
- Moving variable declaration closer to the usage.
- Limit the scope of the try catch block.

Here is a contrived example made to utilise the above points. In the _before_ example, no empty lines are used and we have a try except block wrapping the whole function. As we go through the code, we can see it's parsing the argument into database query parameters, calling the database client to find the fruits and parse the database result to a different representation.

_before_

```python
def search_fruits(supplier_name: str, fruit_name: str) -> List[Fruit]:
  try:
    fruits = []
    query_param = FruitQueryParam(
      supplier_name=supplier_name,
      fruit_name=fruit_name
    )
    rows = database_client.find_fruit(query_param)
    for row in rows:
      fruits.append(Fruit.from_row(row))
    return fruits
  except SupplierDoesNotExists:
    return []
```

Then in the _after_ example, with empty lines, the structure of the function is more apparent upon first glance. We can see clearly there are 3 steps in this function. 

We also moved `fruits = []` to just before iterating the database result rows as that's where we will use `fruits`. This enable us to quickly see what we do with the variable after initialisation. If the initialisation and the usage is far apart, we are forcing the reader to remember how the variable is initialised even though they are in the same function.

Lastly, using the try catch block around `database_client.find_fruit` only. This helps to show where the try catch block is handling and avoids unnecessary indentation on the other lines.

_after_

```python
def search_fruits(supplier_name: str, fruit_name: str) -> List[Fruit]:
  query_param = FruitQueryParam(
    supplier_name=supplier_name,
    fruit_name=fruit_name
  )

  try:
    rows = database_client.find_fruit(query_param)
  except SupplierDoesNotExists:
    return []

  fruits = []
  for row in rows:
    fruits.append(Fruit.from_row(row))
  return fruits
```

**Return early**

In the _before_ example, we have two if else statement blocks validating the basket and fruits before processing them. The actual processing logic is double nested which reduces the maximum line length it can have. Furthermore, we would need to finish reading the long processing logic before reading else block logic even though the else block logic is relatively small. This forces the reader to remember the corresponding if statement so they can understand and validate the else block logic. 

_before_

```python
def pack_fruits(basket, fruits):
  if not basket.is_full():
    if fruits.size() <= basket.capacity():
      # 50 line of codes trying to pack fruits in the basket
    else:
      logger.warning("too much fruit for the basket to hold")
  else:
    logger.warning("basket is full")
```


If we return early in the function, we can improve its clarity. In the _after_ example, the if condition is inverted, once the condition is true, it will trigger the edge case handling logic and exit the function. By using this approach, we can improve the flow of reading the function. Going from top to bottom, the developer read each self-contained blocks of error condition and handling logic then they can move onto to reading the main processing logic. This provides a clear structure of how the function is laid out.

Additionally, returning early helps to reduces indentation of the logic in the function. The processing logic and second if statement block are no longer indented. They can have more space to express each line of logic and more descriptive variable names.

Lastly, it is important to note that returning early is not always beneficial. In a small function or small if-else block, it makes minimal differences as there are relatively less ideas to read and reason.


_after_

```python
def pack_fruits(basket, fruits):
  if basket.is_full():
    logger.warning("basket is full")
    return

  if fruits.size() > basket.capacity():
    logger.warning("too much fruit for the basket to hold")
    return

  # 50 line of codes trying to pack fruits in the basket
```

**High level comments**


When creating a long function, before diving into the implementation. I found it useful to describes each step as comments. For example:

```python
def predict_fruit(image):
  # preprocessing to make the image works with the inference model


  # run the preprocessed image on the inference model


  # post processing on the model result to derive the fruit
```

The comments should be a high level description of what needs to be done and arrange the comments in logical order. For example, having cheap validation step before expensive query or data creation logic to avoid extra computation.

This approach provides opportunities for us to think what is needed in the function. Furthermore, it encourages reasoning about the overall structure of the function upfront. Once we are happy with the high level comments, we can fill in the implementation under each comment.

After the implementation is done, we can remove the comments. In some cases, those comments can becomes a high level summary for the related code block. For instances:

```python
# prepare fruits to be serve on a plate.
for fruit in fruits:
  wash_fruit(fruit)

  slice_fruit(fruit)
```

The comment regarding the for loop allow us to understand what the loop is doing without reading the logic. This increase clarity of the logic if the loop are doing multiple things at once and it's unclear at a glance the overall purpose of the loop.


## Writing Concisely

In English, we want to write concisely so the reader can quickly understand the main points of the sentence. This means having sufficient amount of meaningful words to communicate clearly. The book outlines four tips to help write concisely.

**Strike out useless words** - Remove repetitive words and unnecessary courteous phrase in the sentence. For example, "The cheque that was received from Classic Assurance was received on 13 January". "was received" are repeated so we can remove one of them and the example becomes "The cheque from Classic Assurance came on 13 January".

**Prune the dead wood, grafting on the vigorous** - Replacing unnecessary words with a more effective ones. For example, "The final account dated 28 June from which I note that six payments of 18 euro dollar were credited to our account from 28 March to 25 August, totaling 108 euro dollar. We can replaces "from which I note" with an vigorous verb "show". 

**Shortening wordy prepositional** - Removing preposition such as _for_, _to_, _by_ and _of_. They don't change the meaning of the sentence. For example, "We need approval of the court" to "We need court approval"

**Rewriting completely** - Rewrite the sentence if there are too many useless words in the sentence and applying the previous methods still result in a long sentence with few feeble verbs. To rewrite, we need to understand the main points of the sentence and delivery the most important ideas for the reader in the newer version.

### Coding concisely

Similarly in coding, we want to write concise codes so that the reader can quickly understand. What is concise code? I think it is code that has a clear intended purpose. It satisfies technical and non technical requirements such readability and performance. Furthermore, the current form is the most efficient and effective representation for the logic. This means not necessarily the shortest code like all the logic in a single line but long enough to clearly express the logic in a readable fashion.

The concept of concise code is applicable at every layer of the code base. Ranging from a single function, codes that integrates different modules together and codes that models the data and domain specific concepts. Given the broad scope, I'm limiting the discussion and examples to the codes within a single function.

How can we write concise code at the function level? The above tips for Plain English cut out the redundant part of the sentence and aims to deliver the main points in the most concise and effective way. Deriving from these tips, I can think of two general theme that can help us write code more concisely.

#### Remove codes that aren't serving a purpose

Just like "strike out useless words" and "shortening wordy prepositional", we want to remove any lines of code that doesn't contributes in the function. For example, unused variables, unused arguments and dead code paths. These lines of code often introduce more questions and confusion to the developer when they read the code. These lines distract the developer from building a good mental model of the code.

Fortunately, modern IDEs, code editors and linters are awesome at highlighting these useless lines of code. Some languages like Golang doesn't allow you to compile the program if there are unused variables. There are still some cases where it's not highlighted by the tooling and the line of code doesn't serve a significant purpose. For example:

```python
name = payload["name"]
age = payload["age"]
person = Person(name, age)
```

In the above code snippet, we retrieve values from the payload map and assign them to variables. Then passing the variables to a constructor to create an object. If those variables are not used afterward, then the value assignment is not useful. We can remove the variables and directly use the map retrieval expression to pass to the constructor.

```python
person = Person(payload["name"], payload["age"])
```

There are scenario where value assignment that doesn't add to the functionality but improves the readability. For example, breaking down complex boolean condition.

```python
is_valid = a and (b or c)
is_current = e or d

if is_valid and is_current
```

Grouping the boolean condition with variable assignments provide a way to create semantic meaning for these boolean conditions. This can increases the clarity of the if statement compared to having all the boolean variables in a single condition. Although, this could hide opportunities to simplify the condition but that's depend on how we group the conditions.

When writing code, we should ask ourselves why each line is there. If a line doesn't have a clear purpose, we should remove it.


#### Simplify logic

Sometimes we can restructure the logic or leverage some language features to simplify the logic. Similar to "prune the dead wood, grafting on the vigorous", where we replace a verbose block of code with a simpler version that performs the same functionality. Simplifying the logic helps us to be more concise as we are able to express the logic with fewer lines, keywords or function calls. For example:

```python
result = []
for score in exam_scores:
  if score >= 40:
    result.append("pass")
  else:
    result.append("fail")
```

The above code snippet creates a filter list by iterating through exam scores and conditionally add the element to the result list. This approach works but it's not the most idiomatic python code. We can use list comprehension to construct the list by iterating through `exam_scores` and conditionally choose the values, this avoids calling `.append` after the list is created. 

```python
result = [
  "pass" if score >= 40 else "fail" 
  for score in exam_scores
]
```

Here is another example:

```python
def is_bad_fruit(fruit) -> bool:
  if not fruit.is_sweet or not fruit.has_seeds:
    return True
  else:
    return False
```

The if else structure is useless since we can use the if statement condition as the return value. This simplify the whole function and it's very clear how the return value of this function is determined.

```python
def is_bad_fruit(fruit) -> bool:
  return not fruit.is_sweet or not fruit.has_seeds:
```

Some of these examples can be picked up by linters as they perform static code analysis on the source code. For example, Pylint has a [refactoring checker](https://pylint.pycqa.org/en/2.4/technical_reference/features.html#refactoring-checker) that can make suggestions like using list comprehension or chained comparison. Golangci-lint have a [gosimple](https://github.com/dominikh/go-tools/tree/master/simple) linter that can suggests removing unnecessary if statement checks and using standard library functions.


## Conclusion

Through the examples and discussions, we have defined what is plain code and applied some of the Plain English guidelines to help us write in plain code. To summarise how we apply the guidelines:

- Reducing the amount of mental work can help the developer to understand the code quicker. We can favour positive phrase when suitable and simplify nested conditions.
- There are a lot of existing framework and pattern that we can use to create structure within our software application. We also discussed at the function level, we can arrange the lines of code in logical order, returning early and use high level comment to help us structure logic clearly.
- Tools like linter and IDE can help you to write concise code. They can highlight useless lines of code and make suggestion for simplifications. It's useful to include them as part of your toolkit for development.

Lastly, we merely scratch the idea of plain code. We only discussed about function level logic. In other layers, there are many ways to structure the software application and model the data. Furthermore, the idea of plain code could be different between object oriented programming, functional programming and other programming paradigms. There are researches about coding comprehension and code complexity, it would be insightful to discuss them in respect to plain code. So definitely a lot of ideas and concepts to explore in the future.
