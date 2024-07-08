# COSC320-GroupD

Hello World!


# CropDoc
## Setup Instructions
1. Make sure you're in 'CropDoc' Directory

```cd CropDoc```

2. Create a virtual environment

```python -m venv venv```

3. Activate the environment
    
    ```source venv/bin/activate```(linux)
        
    ```venv\Scripts\activate```(windows)

4. Install Requirements

```pip install -r requirements.txt```


## Running the server
1. Make sure you're in 'CropDoc' Directory

```cd CropDoc```

2. Activate the environment
    
    ```source venv/bin/activate```(linux)
        
    ```venv\Scripts\activate```(windows)

3. Run the server in development mode

```python manage.py runserver```
or
```flask run```




## Maintenance

## Adding a new python library

1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Install the library using pip

```pip install <library-name>```

3. Update the requirements.txt file

```pip freeze > requirements.txt```

4. Commit the changes

```git add requirements.txt```

```git commit -m "Added <library-name>"```

```git push```


## Running formatters over code
1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Run the formatter

```python manage.py format```
