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

Peking University is one of China's top2 universities and are surely worth visiting. In order to provide tourism solutions for tourists and campus security staff, we build PKU-Route-Planner to achieve a no-return tourism solution.

We utilized 数学规划方法 towards three specific usage:
 - Interact with LLMs to provide reliable travel spots.
 - Manually choose spots to visit.
 - For school staff, provide different paths (in this case: 最短路问题和欧拉通路分别建模）.

### 2. Math Model

#### Hamiltonian Path on a Directed Multigraph (有向多重图）

Given a directed multigraph $ G = (V, E) $, where:

- $ V $ is the set of nodes, indexed from $ 0 $ to $ n - 1 $
- $ E \subseteq V \times V \times \text{ID} $ is the set of directed multi-edges, with each edge represented as $ (i, j, \text{id}) $, denoting a directed edge from node $ i $ to node $ j $, identified by a unique edge id
- Each edge $ (i, j, \text{id}) $ has an associated weight (walking time) $ w_{ij}^{(\text{id})} $

##### Decision Variables

- $ x_{ij}^{(\text{id})} \in \{0, 1\} $: Binary variable indicating whether the edge $ (i, j, \text{id}) $ is selected in the path
- $ u_i \in [0, n-1] $: Auxiliary continuous variable representing the relative position of node $ i $ in the path (used for subtour elimination)

##### Objective Function

Minimize the total walking time of the Hamiltonian path:

$$
\min \sum_{(i, j, \text{id}) \in E} w_{ij}^{(\text{id})} \cdot x_{ij}^{(\text{id})}
$$

##### Constraints

1. Degree Constraints

Start node (node 0):

$$
\sum_{(0, j, \text{id}) \in E} x_{0j}^{(\text{id})} = 1 \quad \text{(out-degree)}
$$
$$
\sum_{(i, 0, \text{id}) \in E} x_{i0}^{(\text{id})} = 0 \quad \text{(in-degree)}
$$

End node (node $ n-1 $):

$$
\sum_{(i, n-1, \text{id}) \in E} x_{i,n-1}^{(\text{id})} = 1 \quad \text{(in-degree)}
$$
$$
\sum_{(n-1, j, \text{id}) \in E} x_{n-1,j}^{(\text{id})} = 0 \quad \text{(out-degree)}
$$

Intermediate nodes $ j \in V \setminus \{0, n-1\} $:

$$
\sum_{(i, j, \text{id}) \in E} x_{ij}^{(\text{id})} = 1 \quad \text{(in-degree)}
$$
$$
\sum_{(j, k, \text{id}) \in E} x_{jk}^{(\text{id})} = 1 \quad \text{(out-degree)}
$$

2. Subtour Elimination (Miller–Tucker–Zemlin Constraints)

To prevent disconnected cycles (subtours), we use MTZ constraints:

For all $ (i, j, \text{id}) \in E $, with $ i \ne j $:

$$
u_i - u_j + n \cdot x_{ij}^{(\text{id})} \le n - 1
$$
Where $ u_i \in [0, n-1] $ for all $ i \in V $

