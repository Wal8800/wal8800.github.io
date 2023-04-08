---
layout: post
title:  "Writing plain code"
date:   2023-02-11 17:47:48 +1300
categories: jekyll update
---

Looking to improve the clarity of my technical writing and documentation, I read "Oxford Guide to Plain English". The book provides different guidelines to help people write in Plain English. It defines Plain English as "A communication is in plain language if its wording, structure, and design are so clear that the intended audience can easily find what they need, understand what they find, and use that information." The definition is taken from [Plain Language Association International](https://plainlanguagenetwork.org/plain-language/plain-language-around-the-world/).

I found some of the guidelines in the book to be relevant for coding as developers should easily understand an existing code base. Furthermore, it's interesting to reflect the way I code against these guidelines. Here are some examples:

## Converting negative to positive

The book recommends using positive phrases over negative phrases and only use negative when it is necessary. For example, "Vote for only one candidate only" instead of "Vote for no more than one candidate". For negative phrase, people often need to perform extra mental work as they need to figure out the positive meaning first. By favouring positive phrases, it hopes to reduces the amount of mental work and increases clarity. 

Favouring positive phrases can be directly apply to function and variable names if it's suitable to the application context. More importantly, I think it's the idea of wanting to reduce extra mental work that has wider implication to the code. I often find myself simplifying boolean conditions to attempt to reduce the extra mental work for the reader when they evaluate the boolean condition.

For example, if you have a nested boolean condition:


```python
not (fruit.is_sweet and fruit.has_seeds)
```

We can apply [DeMorgan Laws](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) and the condition becomes


```python
not fruit.is_sweet or not fruit.has_seeds
```

Then the reader no longer needs to evaluate the bracket first, they can evaluate the condition as they read from left to right.

Another scenario that I observed that could use a review is checking if a boolean field is false.

```python
not config.disabled
```

In almost all cases, the condition is the simplest and it's clear to the reader. However, if we need to repeatedly check the same conditions, it makes me wonder if we should invert the boolean variable. For example:

```python
config.enabled
```

This means we don't need to attach a logical not operator in each of the conditions and we directly use the semantic meaning of the field in the condition i.e. if enable then do this.

Interestingly, this is not a change that I would recommend or make frequently. It's something I would think about when adding new argument or creating new function. Changing an existing code base can lead a great number of changes if there are a lot of dependencies. The minor benefits doesn't out weigh the cost especially if we need to make breaking changes to an existing API. 


## Writing short sentences and clear paragraph 

Long sentences are relatively harder to read as they presents more ideas and require more short term memory. The book recommends to keep sentences short and presents 1 main idea at a time. Alternatively, format the long sentence into a vertical list for clearer presentation of multiple ideas.

In coding, a lot of style guides and formatters have maximum line length. Python PEP-8 standard recommends 79 characters, Javascript Prettier code formatter recommends 80 characters and Google's Java style guide recommends 100 characters. Some development teams also tend to agree on a higher limit such as 120 characters on private codes that they maintained.

The maximum line length helps to reduce the amount of logic squeezed onto a single line, encouraging the developer to break down the logic into multiple lines. In addition, it highlights any excessively long variable name or function name as they can make the line go over the maximum length. The maximum length also improves the experience of viewing multiple files side by side as the lines can avoid line wrap that disrupts the original visual structure. This is helpful when comparing code such as code review.

// example image


Next, a collection of sentences forms a paragraph. To write a clear paragraph, the ideas from the sentences should flow coherently. This can be done by providing a structure. For example, starting with a topic sentence, then examples to the topics and ending with a question or reassurance to the reader. 

Structure exists in various levels of a code base. Furthermore, there are a lot of existing framework, paradigm and methodology that helps developer to implement structure in their code. This means a lot can be discussed about structure in coding. As a result, to limit the scope of the discussion, I will focus on the function level structure only.

When a developer reads a function, the logic in the function needs to be clear so they can understand what the function is doing. This is especially important when the function is long and contains various different logic. In addition, functions can have a variety of different structures depending on their purposes. Over the years, I found the following ideas to be helpful for structuring and improving the clarity of the function.


**Organise the lines in the function into logical blocks**

We can rearrange lines in the function by:

- Using whitespace for separating logic.
- Moving variable declaration closer to the usage.
- Limit the scope of the try catch block.

Here is a contrived example made to utilise the above points. In the _before_ example, no empty lines are used and we have a try except block wrapping the whole function. As we go through the code, we can see it's parsing the argument into database query parameters, calling the database client to find the fruits and parse the database result to a different representation. 

Then in the _after_ example, with empty lines, the structure of the function is more apparent upon first glance. We can see clearly there are 3 steps in this function. 

We also moved `fruits = []` to just before iterating the database result rows as that's where we will use `fruits`. This enable us to quickly see what we do with the variable after initialisation. If the initialisation and the usage is far apart, we are forcing the reader to remember how the variable is initialised even though they are in the same function.

Lastly, using try catch block around `database_client.find_fruit` only. This helps to show where the try catch block looking handle and it avoids unnecessary indentation on the other lines.

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

In the _before_ example, we have two if else statement blocks validating the basket and fruits before processing them. The actual processing logic is double nested which reduces the maximum line length it can have. Furthermore, we would need to finish reading the long processing logic before reading else block logic even though the else block logic is relatively small. This forces the reader to remember the corresponding if statement or scroll back up so they can understand if the else block logic makes sense. 

If we return early in the function, we can improve its clarity. In the _after_ example, the if condition is inverted, once the condition is true, it will trigger the edge case handling logic and exit the function. By using this approach, we can improve the flow of reading the function. Going from top to bottom, the developer read each self contained blocks of error condition and handling logic first then they can move onto to reading the main processing logic. This provides a clear structure of how the function is laid out.

Additionally, returning early helps to reduces indentation of the logic in the function. The processing logic and second if statement block are no longer indented. They can have more space to express each line of logic and more descriptive variable names.

Lastly, it is important to note that returning early is not always beneficial. In a small function or small if-else block, it makes minimal differences as there are relatively less ideas to read and reason.

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
  # preprocessing to make the image fit for the inference model


  # run the preprocessed image on the inference model


  # post processing on the model result to derive the fruit
```

The comments should be high level description of what needs to be done and we can arrange the comments in logical order. For example, having cheap validation step before expensive query or data creation logic to avoid extra work. 

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

We want to write concisely so the reader can quickly understand the main points of the sentence. This means having sufficient amount of meaningful words to communicate clearly. The book outlines four tips to help write concisely.

**Strike out useless words**

Remove repetitive words and unnecessary courteous phrase in the sentence. For example, from the book, "The cheque that was received from Classic Assurance was received on 13 January". "was received" are repeated so we can get remove one of them and the example becomes "The cheque from Classic Assurance came on 13 January".

**Prune the dead wood, grafting on the vigorous**

Replacing unnecessary words with a more effective one. For example, "The final account dated 28 June from which I note that six payments of 18 euro dollar were credited to our account from 28 March to 25 August, totaling 108 euro dollar. We can replaces "from which I note" with an vigorous verb "show". 

**Shortening wordy prepositional**

Removing preposition such as _for_, _to_, _by_ and _of_. They don't change the meaning of the sentence. For example, "We need approval of the court" to "We need court approval"

**Rewriting completely**

Rewrite the sentence if there are too many useless words in the sentence and applying the previous methods still result in long sentence with few feeble verbs. To rewrite, we need to understand the main points of the sentence and delivery the most important ideas for the reader in the newer version.

### Coding concisely

Similarly in coding, we want to write concise codes so that the reader can quickly understand. What is concise code? From my experiences, I think it is code that has a clear intended purpose. It satisfies technical and non technical requirements such readability and performance. Furthermore, the current form is the most efficient and effective representation for the logic. This means not necessarily the shortest code i.e. 1 liner but long enough to clearly express the logic in a readable fashion.

The concept of concise code is applicable at every layer of the code base. Ranging from a single function, codes that integrates different modules together and codes that models the data and domain specific concepts. There could be some redundant and repetitive code in different area of the code base.

Given the broad scope, I'm limiting the discussion and examples to the codes within a single function because the blog post will be very long if we were to discuss the other scenarios.

How can we write concise code within a function? The above tips for English cut out the redundant part of the sentence and aims to deliver the main points in the most concise and effective way. Deriving from these tips, I can think of two general approach to help us write code more concisely. Let's have a look at each one with some examples.

#### Remove codes that aren't serving a purpose

Just like "strike out useless words" and "shortening wordy prepositional", we want to remove any lines that doesn't contributes in the function. For example, unused variables, unused arguments and dead code paths. These lines of code often introduce more questions and confusion for the developer when they read the code as they need to figure out the purpose of these lines. These lines distract the developer from building a good mental model of the code.

Fortunately, modern IDEs, code editors and linters are awesome at highlighting these useless lines of code. Some languages like Golang doesn't allow you to compile the program if there are unused variables. However, there are some cases where it's not highlighted by the tooling and the line of code doesn't serve a significant purpose. For example:

```python
name = payload["name"]
age = payload["age"]
person = Person(name, age)
```

In the above code snippet, we retrieve values from the payload map and assign them to variables. Then passing the variable to constructor to create an object. If those variables are not used afterward, then the value assignment is not useful. We can remove the variables and directly use the map retrieval expression to pass to the constructor.

```python
person = Person(payload["name"], payload["age"])
```

There are scenario where value assignment that doesn't add to the functionality but improves the readability. For example, breaking down complex boolean condition.

```python
is_valid = a and (b or c)
is_current = e or d

if is_valid and is_current
```

Grouping the boolean condition with variable assignments provide a way to create semantic meaning for these boolean conditions. This can increases the clarity of the if statement compared to having all the boolean variables in a single if statement condition. Although, this could hide opportunities to simplify the condition but that's depend on how we group the conditions.

I think when writing code, we should ask why each line is there. If a line doesn't have a clear purpose, we should remove it.


#### Simplify logic

Sometimes when we can restructure the logic or leverage some language features to simplify the logic. This is similar to "prune the dead wood, grafting on the vigorous", where we replace a verbose block of code with a simpler version that performs the same functionality. Simplifying the logic helps us to be more concise as we are able to express the logic with fewer lines, keywords or function calls. For example:

```python
result = []
for score in exam_scores:
  if score >= 40:
    result.append("pass")
  else:
    result.append("fail")
```

The above code snippet creates a filter list by iterating through exam scores and conditionally add the element the result list. This approach works but it's not the most idiomatic python code. We can use list comprehension to construct the list by iterating through `exam_scores` and conditionally choose the values, this avoids calling `.append` after the list is created. 

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

The if else structure is useless since we can use the if statement condition as the return value.

```python
def is_bad_fruit(fruit) -> bool:
  return not fruit.is_sweet or not fruit.has_seeds:
```

Some of these scenarios can be picked up by linters as they perform static code analysis on the source code. For example, Pylint has a [refactoring checker](https://pylint.pycqa.org/en/2.4/technical_reference/features.html#refactoring-checker) that can make suggestions like using list comprehension or chained comparison. Golangci-lint have a [gosimple](https://github.com/dominikh/go-tools/tree/master/simple) linter that can suggests removing unnecessary if statement checks and using standard library functions.


## Conclusion

- the bottom line is the functional behaviour should stay the same though
  - the assignment is needed to reference the value/pointer to a variable
  - using syntax sugar feat has performance implications

- Sometime it can be a bit subjective and needs to agree on within the team.


// linter /static checker will can catch. automated. in ci build
// Lastly, modern IDE are great at highlighting unused variable assignment and dead code paths.

// research about programmer prefer verbosity of the code ??

// In this post, I felt I merely scratch the surface of topic
// There are probably research about developer coding comprehension, readability

