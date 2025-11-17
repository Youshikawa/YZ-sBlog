---
layout: post
title: "[Challenge] Logistics Routing with Port Congestiont"
date: 2025-11-18
categories: [problem]
tags: [optimization]
---

# \[Challenge] Logistics Routing with Port Congestion

## Background

A logistics company must ship a large amount of cargo from Port A to Port B over several days.
Unlike previous operations, the company now wants to account not only for transportation cost and route changes, but also for **congestion risk on ports**.

You are asked to design a multi-day routing plan that keeps the **overall cost as low as possible**.
This is a **challenge / scoring problem**: you are not required to find a provably optimal solution.

---

## 1. Problem Description

* There are **$n$ days**.
* There are **$m$ ports**, numbered $1, 2, \dots, m$.
* Port A is port **1**, and Port B is port **$m$**.
* There are **$e$** undirected shipping routes (edges). Each route connects two ports and has a positive integer length (distance).
* On some days, certain ports are **closed** (unavailable). On such a day, your route is not allowed to pass through those ports.
* On **each day** $t$ ($1 \le t \le n$), you must choose **one route (a path)** from port 1 to port $m$.

Your goal is to choose a valid path for each day so that the **total cost** (defined below) is as small as possible.

---

## 2. Daily Path and Validity

For each day $t$ ($1 \le t \le n$), you must choose a path

$$
P_t = (v_{t,1}, v_{t,2}, \dots, v_{t,L_t})
$$

satisfying:

1. $v_{t,1} = 1$ and $v_{t,L_t} = m$ (start at port 1, end at port $m$);
2. For every $1 \le i < L_t$, there is an undirected edge between $v_{t,i}$ and $v_{t,i+1}$ in the graph;
3. Every port $v_{t,i}$ on the path must be **open** on day $t$ (not closed on that day);
4. You may assume your path is simple (no repeated nodes), though this is not enforced by the statement.

Let the **length of the path** on day $t$ be

$$
len_t = \sum_{i=1}^{L_t - 1} w(v_{t,i}, v_{t,i+1})
$$

where $w(u, v)$ is the given length of edge $(u,v)$.

---

## 3. Cost Components

The total cost of your multi-day routing plan consists of **three parts**:

1. Transportation distance cost
2. Route change cost
3. Port congestion cost

Your objective is to minimize the **TotalCost**, defined below.

### 3.1 Transportation Distance Cost

This is simply the sum of daily path lengths:

$$
Cost_{\text{dist}} = \sum_{t=1}^n len_t
$$

### 3.2 Route Change Cost

We assume that on day 1, the system selects an initial route; this **does not** count as a change.

For each day $t \ge 2$, if the path on day $t$ is **different** from the path on day $t-1$, we count **one route change**.

Formally:

* Let $chg$ be the number of days $t$ ($2 \le t \le n$) such that $P_t \neq P_{t-1}$.
* Here, $P_t \neq P_{t-1}$ means the sequences of vertices in the two paths are not exactly the same.

You are given an integer parameter $k$, the cost of changing the route once. Then:

$$
Cost_{\text{change}} = k \cdot chg
$$

### 3.3 Port Congestion Cost (Key Novel Part)

For each port $p$ ($1 \le p \le m$), define its **usage count** over all days:

$$
use_p = 
\big|\{ t \in \{1,\dots,n\} \mid p \text{ appears somewhere in path } P_t \}\big|
$$

That is, the number of days on which your route passes through port $p$.

You are given a parameter $\lambda$.
The **port congestion cost** is defined as:

$$
Cost_{\text{cong}} = \lambda \cdot \sum_{p=1}^m use_p^2
$$

Intuition:

* If a port is used on only a few days, $use_p$ is small, so its squared penalty is small.
* If a port is heavily used across many days, $use_p^2$ becomes large, significantly increasing the cost.
* This encourages you to **distribute traffic** across different ports, rather than overloading a few.

---

## 4. Total Cost (Objective Function)

Combining the three parts:

$$
\boxed{
TotalCost = Cost_{\text{dist}} 
           + Cost_{\text{change}} 
           + Cost_{\text{cong}}
}
$$

That is,

$$
\boxed{
TotalCost = 
\left(\sum_{t=1}^n len_t\right)
+ k \cdot chg
+ \lambda \cdot \sum_{p=1}^m use_p^2
}
$$

Your task is to choose valid paths $P_1, P_2, \dots, P_n$ to **minimize** this TotalCost.

> Note: This is a **challenge / scoring problem**.
> The judge computes your TotalCost based on your output, and lower is better.
> You are not required to find a provably optimal solution.

---

## 5. Input Format

Input is given via standard input in the following format:

1. First line: five integers

   $$
   n,\ m,\ k,\ \lambda,\ e
   $$

   where:

   * $n$: number of days;
   * $m$: number of ports (port 1 is A, port $m$ is B);
   * $k$: route change cost;
   * $\lambda$: congestion cost parameter;
   * $e$: number of shipping routes (edges).

2. Next $e$ lines: each contains three integers

   $$
   u,\ v,\ w
   $$

   meaning there is an undirected route between port $u$ and port $v$ with length $w$.

3. Next line: one integer $d$, the number of “port closed” intervals.

4. Next $d$ lines: each contains three integers

   $$
   p,\ a,\ b
   $$

   meaning port $p$ is **closed** (unavailable) on days from $a$ to $b$ (inclusive), i.e.,
   on all days $t$ with $a \le t \le b$, path $P_t$ may **not** use port $p$.

You may assume that:

* For every day $t$ ($1 \le t \le n$), there exists **at least one valid path** from port 1 to port $m$ using only ports that are open on day $t$.

---

## 6. Output Format 

You must output a **complete routing plan**, i.e., one valid path per day.

Output exactly **$n$ lines**.
On the $t$-th line, describe the path $P_t$ for day $t$ as:

$$
L_t\ v_{t,1}\ v_{t,2}\ \dots\ v_{t,L_t}
$$

where:

* $L_t$ is the number of vertices in the path on day $t$;
* You must satisfy:

  * $v_{t,1} = 1$ and $v_{t,L_t} = m$;
  * For all $1 \le i < L_t$, there is an edge between $v_{t,i}$ and $v_{t,i+1}$;
  * For every vertex $v_{t,i}$, port $v_{t,i}$ must be open on day $t$ (not within a closed interval for that port).

The judge will:

1. Verify that each daily path is valid.
2. Compute:

   * Each $len_t$,
   * The number of route changes $chg$,
   * Each $use_p$,
   * Then calculate your TotalCost.

If your output format is invalid, or any path is invalid for its day (disconnected, wrong endpoints, uses closed ports, etc.), the score for that test may be 0 (exact behavior depends on the contest system).

---

## 7. Scoring 


* For each test file $i$:

  * If your solution is invalid, you get **0** for that test.
  * Otherwise, the judge computes your $TotalCost_i$.
* Let $Baseline_i$ be the smallest TotalCost among all contestants (and official solutions) on test $i$.
* Your score on test $i$ may be defined as, for example:

$$
Score_i = 100000 \times \frac{Baseline_i}{TotalCost_i}
$$

* Your final score is the average (or weighted sum) of all $Score_i$.

The exact normalization formula may be different; the **objective function** and **validity rules** are as specified above.

---

## 8.Constraints


* $1 \le n \le 200$
* $2 \le m \le 50$
* $1 \le e \le 400$
* $1 \le k, \lambda \le 10^4$
* $1 \le w \le 10^4$
* $1 \le d \le 2000$
* For every day $t$, there exists at least one valid path from 1 to $m$.

---

## 9. Sample (Illustrative Only)

This is a **very small** example, only to illustrate input and output formats.
The sample output is **not guaranteed optimal**.

### Sample Input

```text
3 4 5 2 4
1 2 1
2 4 1
1 3 2
3 4 2
2
2 2 2
3 3 3
```

Explanation:

* 3 days, 4 ports (1 is start, 4 is end).
* Route change cost $k = 5$, congestion parameter $\lambda = 2$.
* Edges:

  * 1–2 with length 1
  * 2–4 with length 1
  * 1–3 with length 2
  * 3–4 with length 2
* Closed intervals:

  * Port 2 is closed on day 2;
  * Port 3 is closed on day 3.

Possible valid paths:

* Day 1: can use 1–2–4 or 1–3–4
* Day 2: port 2 is closed → must use 1–3–4
* Day 3: port 3 is closed → must use 1–2–4

### Sample Output (valid but not necessarily optimal)

```text
3 1 2 4
3 1 3 4
3 1 2 4
```

Explanation:

* Day 1: path 1-2-4, length $1 + 1 = 2$
* Day 2: path 1-3-4, length $2 + 2 = 4$
* Day 3: path 1-2-4, length $1 + 1 = 2$

1. Distance cost:

   $$
   Cost_{\text{dist}} = 2 + 4 + 2 = 8
   $$

2. Route change cost:

   * Day 1 vs Day 2: different → 1 change
   * Day 2 vs Day 3: different → 1 change
   * Total $chg = 2$

   $$
   Cost_{\text{change}} = 5 \times 2 = 10
   $$

3. Congestion cost:

   * Port 1: used on all 3 days → $use_1 = 3$
   * Port 2: used on days 1 and 3 → $use_2 = 2$
   * Port 3: used on day 2 → $use_3 = 1$
   * Port 4: used on all 3 days → $use_4 = 3$

   $$
   \sum use_p^2 = 3^2 + 2^2 + 1^2 + 3^2 = 9 + 4 + 1 + 9 = 23
   $$

   $$
   Cost_{\text{cong}} = \lambda \cdot 23 = 2 \times 23 = 46
   $$

4. Total cost:

   $$
   TotalCost = 8 + 10 + 46 = 64
   $$

The judge would use this TotalCost value during scoring.

