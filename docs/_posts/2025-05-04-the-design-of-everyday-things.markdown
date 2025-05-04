---
layout: post
title: "The Design of Everyday Things in Software Development"
date:   2024-10-09 17:00:48 +1300
---

**The Design of Everyday Things** is a book about how humans interact with products, offering principles and frameworks that help improve user experience.
What has stuck with me the most is how these principles appear not just in the end-user experience of a software product, like a web or mobile app, but throughout broader system design and the entire development life cycle. It is fascinating to see how software systems apply these ideas to improve usability for everyone involved, including developers, operators, and users. Here are some of the principles:

## Constraints are powerful clues

Applying constraints can make a product’s function easier to discover and use. They can also make feedback clearer. Different constraints appear across various parts of software systems.

### Code
In statically typed languages, type constraints help communicate the intended use of functions and variables. For example, if a function returns a nullable string (String?), it signals to the developer that they need to handle the null case. Ignoring this could lead to runtime errors. These constraints not only reduce bugs but also improve readability and maintainability.

### System Architecture
When a software system enforces that all external API requests go through a single API gateway, it creates a logical constraint on how requests flow into the system. This results in a simpler mental model compared to systems with multiple entry points. It also makes it easier for developers to trace incoming traffic, since there is only one place to look.

### Tooling and Tech Stack
Standardizing the tech stack, such as using React for frontend, Express for backend, and PostgreSQL for the database, reduces cognitive load for developers moving between projects. For example, if every internal web app uses the same component library, a developer switching teams can contribute more quickly. This is because UI patterns are familiar and tooling, such as linters and build pipelines, behaves predictably.


## Designing for errors
In the book, Norman discusses different types of user errors and how to design products that help users avoid or recover from them. This idea also applies to software systems, where errors are inevitable. What matters is how we handle them. Many modern best practices are built around this principle.

### Failing safely and visibly
In a web application, showing a clear error message like “Invalid email format” helps users recover quickly. In backend systems, failing fast with descriptive logs, rather than silently failing, helps engineers detect and resolve issues sooner.

### Minimizing time to recovery
In distributed systems, patterns like circuit breakers help isolate failures and prevent cascading problems. If one service becomes unresponsive, the circuit breaker trips and returns a fallback response. This allows the system to degrade gracefully instead of crashing completely.

### Proactive detection
Observability tools like Prometheus and Grafana, paired with alerting systems like PagerDuty, help detect issues before users even notice them. For example, an alert might trigger when API latency exceeds a certain threshold. This gives engineers a chance to respond early and avoid user-visible problems.

## People and technology
The book argues that combining people and technology creates more effective systems than relying on either alone. This insight is especially relevant today, with the rapid rise of AI and automation.
Machines excel at consistency and precision. Humans contribute creativity, adaptability, and context. Systems that support human strengths and mitigate human limitations tend to produce better outcomes.

In code review processes, static analysis tools like ESLint or SonarQube catch syntax or style issues. This allows human reviewers to focus on deeper concerns, such as architecture, logic, and user impact. The result is higher code quality than either approach could achieve alone.

AI tools like GitHub Copilot enhance developer productivity by generating boilerplate or suggesting solutions. Developers, however, still guide the direction, review the output, and ensure that the results align with business needs.

## Final thoughts
The principles from The Design of Everyday Things are not just for product designers. They are valuable for anyone involved in building software. Whether the goal is to make code easier to read, systems easier to debug, or tools easier to use, thoughtful design benefits everyone. When we apply these ideas consistently across the development lifecycle, we improve not only the systems we build but also the experience of working with them.
