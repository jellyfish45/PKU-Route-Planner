# PKU-Route-Planner

## 1. Project Structure

```
project/
├── Hamilton.py
├── Euler.py                    # Solvers
├── utils.py                    # useful functions
├── Render.py                   # Web drawing
├── app.py                      # Streamlit main program
├── data/
│   ├── pku_all_simple_paths.csv   
│   └── pku_locations_updated.csv    
└── requirements.txt           # Dependencies required
```


## 2. Installation

Download "pku_all_simple_paths.csv" via [Google Drive](https://drive.google.com/file/d/1gJPiplxBd81p2kiZ-dioGI8kDJ5DVMLb/view?usp=drive_link) first. 
```python
git clone https://github.com/jellyfish45/PKU-Route-Planner.git
cd PKU-Route-Planner
pip install -r requirements
streamlit run app.py
```
