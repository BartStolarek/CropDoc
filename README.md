# COSC320-GroupD

Hello World!


# Setup Instructions
1. Clone the repository
2. Create a virtual environment

```python -m venv venv```

3. Activate the environment
    
    ```source venv/bin/activate```(linux)
        
    ```venv\Scripts\activate```(windows)

4. Install Requirements

```pip install -r requirements.txt```




# Maintenance

## Adding a new python library
1. Install the library using pip

```pip install <library-name>```

2. Update the requirements.txt file

```pip freeze > requirements.txt```

3. Commit the changes

```git add requirements.txt```

```git commit -m "Added <library-name>"```

```git push```