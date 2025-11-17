---
layout: post
title: "[Challenge]Designing a Tempest"
date: 2025-11-17
categories: [problem]
tags: [optimization]
---
## Problem: *Designing a Tempest*

You live on an infinite 2D grid of square cells.

Each cell has a color: **black** or **white**.

Initially, all cells $(x, y)$ with

$$
0 \le x \le X - 1,\quad 0 \le y \le Y - 1
$$

are black, and all other cells are white.
So the initial black region is a solid $X \times Y$ rectangle.

You may design exactly $N$ *great storms*.
Each storm transforms the entire grid according to parameters that you choose.

Your goal is to choose these storms so that **after all $N$ storms**, the number of connected components of black cells is as large as possible.

This is an optimization (heuristic) problem: there is no fixed “correct output”; instead, your solution is graded by how good its storms are.

---

## 1. Great storms

You must define **exactly $N$** great storms.

The $i$-th storm is defined by a character $C_i$ and two integers $A_i, B_i$:

* $C_i$ is either `'X'` or `'Y'`;
* $A_i, B_i$ are integers satisfying

  $$
  -1000 \le A_i, B_i \le 1000.
  $$

### 1.1 Effect of one storm

We describe the update in terms of **where the color comes from**.

Let the color *before* the $i$-th storm be given.
After the storm, the color of each cell $(x, y)$ is taken from some cell *before* the storm, as follows.

#### Case $C_i = \text{'X'}$

* If $x < A_i$, then the new color of $(x, y)$ is the old color of $(x, y + B_i)$.
* If $x \ge A_i$, then the new color of $(x, y)$ is the old color of $(x, y - B_i)$.

#### Case $C_i = \text{'Y'}$

* If $y < A_i$, then the new color of $(x, y)$ is the old color of $(x + B_i, y)$.
* If $y \ge A_i$, then the new color of $(x, y)$ is the old color of $(x - B_i, y)$.

This update is applied simultaneously to all cells on the infinite grid.

It can be shown that this mapping is a bijection on cells, so **the number of black cells is always exactly $X \times Y$**.

---

## 2. Adjacency and connected components

Two black cells $(x_1, y_1)$ and $(x_2, y_2)$ are **adjacent** if and only if

$$
|x_1 - x_2| + |y_1 - y_2| = 1
$$

(that is, they share a side).

A black cell can move to another black cell by repeatedly moving to adjacent black cells.

Two black cells are **connected** if and only if one can move from one to the other by such moves.

A non-empty set $S$ of black cells is a **connected component** if:

* Any two cells in $S$ are connected;
* No black cell outside $S$ is connected to any cell in $S$.

After all $N$ storms have been applied, suppose the black cells form $K$ connected components.
Let the size of the $j$-th component (number of cells in it) be $s_j$. Then

$$
\sum_{j=1}^{K} s_j = X \times Y.
$$

---

## 3. Objective and scoring

For each test case, your **objective** is to maximize $K$, the number of connected components of black cells after all storms.

### 3.1 Score for one test

For a single test case with parameters $(N, X, Y)$, after simulating your $N$ storms and computing $K$, the score is

$$
\text{score} = 10^6 \times \frac{K}{X \times Y}.
$$

* If all black cells remain in a single component ($K = 1$), the score is about $10^6 / (X Y)$, very small.
* If every black cell is isolated ($K = X Y$), the score is exactly $10^6$, which is the theoretical maximum for that test.

### 3.2 Final contest score

The input contains multiple test cases.
Your **final score** is the arithmetic mean of the scores over all test cases, rounded down to an integer.

Higher is better.

---

## 4. Input and output format

### 4.1 Input

The judge gives your program:

* First line: an integer $T$ — the number of test cases.
* Then for each test case:

  * One line with three integers: $N, X, Y$.

Example (single test):

```text
1
3 5 7
```

#### Constraints (example setting)

These are example limits suitable for a heuristic contest; they can be tuned by the contest organizers:

* $1 \le T \le 50$
* $1 \le N \le 30$
* $5 \le X, Y \le 50$

### 4.2 Output

For **each** test case, you must print exactly $N$ lines.

The $i$-th line should contain:

```text
C_i A_i B_i
```

where:

* `C_i` is `X` or `Y`;
* `A_i` and `B_i` are integers with $-1000 \le A_i, B_i \le 1000$.

Example output for the sample input above:

```text
X 0 1
Y 2 -1
X 1 2
```

There is no blank line between test cases.

If you want a storm that has little effect, you may choose parameters that roughly “cancel out” (for example, $B_i = 0$, or an $A_i$ far away from the current black region).

---

## 5. Implementation notes for the judge (informal)

(This is for clarity; participants don’t need to implement the judge.)

For each test case:

1. Represent the black region as a union of axis-aligned rectangles.
   Initially, it is a single rectangle $[0, X-1] \times [0, Y-1]$.

2. For each storm $(C_i, A_i, B_i)$:

   * For each current rectangle, cut it by the line $x = A_i$ (if $C_i = X$) or $y = A_i$ (if $C_i = Y$),
     obtaining at most two rectangles;
   * Translate each side up/down or left/right according to the rules above;
   * Collect all resulting rectangles.

   It can be shown that after each storm the region is still a union of rectangles, and the number of rectangles stays manageable for the given bounds.

3. After all storms, build a graph whose nodes are rectangles;
   connect two nodes if their rectangles contain at least one pair of adjacent cells.
   The connected components of this graph correspond exactly to connected components of the black cells.

4. The size of a component is the sum of areas of rectangles in it.
   Count components and compute the score.

