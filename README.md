# PKU-Route-Planner

## Preparation

### 1. Project Structure

```
project/
├── Hamilton.py
├── Euler.py                    # Solvers
├── utils.py                    # useful functions
├── Render.py                   # Web drawing
├── app.py                      # Streamlit main program
├── data/
│   ├── pku_all_simple_paths.csv   
│   ├── pku_locations_updated.csv
│   └── pku_walk_node_locations.csv
└── requirements.txt           # Dependencies required
```


### 2. Installation

Download "pku_all_simple_paths.csv" via [Google Drive](https://drive.google.com/file/d/1gJPiplxBd81p2kiZ-dioGI8kDJ5DVMLb/view?usp=drive_link) first. After sucessful installation of the project, put it in **data** file.
```python
git clone https://github.com/jellyfish45/PKU-Route-Planner.git
cd PKU-Route-Planner
pip install -r requirements
streamlit run app.py
```

## Project Backgroud

### 1. Reality Backgroud

Peking University is one of the top two universities in China and is undoubtedly worth visiting. To support both tourists and campus security staff, we have developed PKU Route Planner — a solution designed to generate non-repetitive (no-return) travel routes across campus.

We apply mathematical programming techniques to address three key use cases:
- LLM-assisted recommendations for popular and meaningful spots to visit.
- Manual selection of points of interest for customizable route planning.
- Path planning for campus staff, including tailored solutions using models for the shortest path and the Eulerian trail problem.

### 2. Math Model

#### Hamiltonian Path on a Directed Multigraph (有向多重图）

Given a directed multigraph$G = (V, E)$, where:

-$V$is the set of nodes, indexed from$0$to$n - 1$
-$E \subseteq V \times V \times \text{ID}$is the set of directed multi-edges, with each edge represented as$(i, j, \text{id})$, denoting a directed edge from node$i$to node$j$, identified by a unique edge id
- Each edge$(i, j, \text{id})$has an associated weight (walking time)$w_{ij}^{(\text{id})}$

##### Decision Variables

-$x_{ij}^{(\text{id})} \in \{0, 1\}$: Binary variable indicating whether the edge$(i, j, \text{id})$is selected in the path
-$u_i \in [0, n-1]$: Auxiliary continuous variable representing the relative position of node$i$in the path (used for subtour elimination)

##### Objective Function

Minimize the total walking time of the Hamiltonian path:

$$
\min \sum_{(i, j, \text{id}) \in E} w_{ij}^{(\text{id})} \cdot x_{ij}^{(\text{id})}
$$


#### Eulerian Path on a Undirected Simple Graph（无向简单图）


Given an undirected simple graph$G = (V, E)$, where:

-$V$is the set of vertices;
-$E$is the set of undirected edges (no self-loops or multiple edges).

The goal is to determine whether there exists an Eulerian path that traverses each edge **exactly once**.

##### Decision Variables

-$x_{ij} \in \{0,1\}$: Equals 1 if edge$(i, j)$is selected in the path, 0 otherwise.
-$f_{ij} \in \mathbb{Z}_{\geq 0}$: Flow from node$i$to node$j$, used to eliminate subtours.
-$y_i \in \{0,1\}$: Equals 1 if node$i$is selected as the starting point.
-$z_i \in \{0,1\}$: Equals 1 if node$i$is selected as the ending point.


##### Objective Function

Maximize:
$$
\max\ 1
$$

(This is a feasibility problem — the goal is simply to determine whether a feasible Eulerian path exists.)


##### Constraints

1. **Flow Capacity Constraints (simplified MTZ-style subtour elimination)**  
  $$
   f_{ij} + f_{ji} \leq |V| - 1 \quad \forall (i,j) \in E
  $$

2. **Flow Conservation at the Start Node$s$**  
  $$
   \sum_{j \in N(s)} (f_{sj} - f_{js}) = |V| - 1
  $$

3. **Flow Balance for All Other Nodes**  
  $$
   \sum_{j \in N(i)} (f_{ij} - f_{ji}) = -1 \quad \forall i \in V \setminus \{s\}
  $$

4. **Non-negativity of Flow Variables**  
  $$
   f_{ij} \geq 0 \quad \forall i \neq j \in V
  $$

5. **Degree Constraints (Path Structure)**  
  $$
   \sum_{j \in N(i)} (x_{ij} - x_{ji}) = y_i - z_i \quad \forall i \in V
  $$

6. **Unique Start and End Nodes**  
  $$
   \sum_{i \in V} y_i = 1,\quad \sum_{i \in V} z_i = 1
  $$

7. **Undirected Edge Constraint (Simple Graph)**  
  $$
   x_{ij} + x_{ji} \leq 1 \quad \forall (i,j) \in E
  $$

8. **Variable Domains**  
  $$
   x_{ij} \in \{0,1\} \quad \forall (i,j) \in E \\
   y_i, z_i \in \{0,1\} \quad \forall i \in V
  $$
