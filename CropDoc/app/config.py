import os

app_directory = os.path.abspath(os.path.dirname(__file__))

# Find the server/apps parent directory
check_dir = app_directory
while not os.path.exists(os.path.join(check_dir, 'app')):
    check_dir = os.path.dirname(check_dir)
root_project_dir = check_dir


# Find the config env file
def find_config_env(start_dir):
    current_dir = os.path.abspath(start_dir)
    
    while current_dir:
        config_env_path = os.path.join(current_dir, 'config.env')
        if os.path.isfile(config_env_path):
            return config_env_path
        
        # Move to the parent directory
        parent_dir = os.path.dirname(current_dir)
        
        # If current directory is the same as parent directory, break the loop
        if current_dir == parent_dir:
            break
        
        # Update current directory to parent directory for next iteration
        current_dir = parent_dir
    
    return None

# Import the file
def import_config_env(config_env_path):
    print(f'Importing environment from .env file from {config_env_path}')
    for line in open(config_env_path):
        var = line.strip().split('=')
        if len(var) == 2:
            os.environ[var[0]] = var[1].replace("\"", "")

        
config_env_path = find_config_env(app_directory)

# Look for config.env on the same level as config.py
if config_env_path is not None and os.path.exists(config_env_path):
    import_config_env(config_env_path)
else:
    # Check if its in root
    config_env_path = os.path.join(root_project_dir, 'config.env')
    if os.path.exists(config_env_path):
        print('config.env file not found in app directory, but found in root directory. Advise to move it to app directory.')
        print('Importing environment from .env file')
        import_config_env(config_env_path)
    else:
        print(
            'config.env file not found, please read README.md for config.env file structure'
        )
        
        
        
class AppConfig:
    APP_NAME = os.environ.get('APP_NAME', 'App_Name_Missing')
    if os.environ.get('SECRET_KEY'):
        SECRET_KEY = os.environ.get('SECRET_KEY')
    else:
        SECRET_KEY = 'Secret_Key_Missing'
        print('Secret key not set, please set it in config.env file')

    # Flask Config
    FLASK_ENV = os.environ.get('FLASK_ENV', 'default')

    # Log Level
    LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', 'DEBUG')

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(AppConfig):
    ENV = 'development'
    DEBUG = True
    LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', 'DEBUG')
    
    @classmethod
    def init_app(cls, app):
        print('Development Environment Activated.')


class ProductionConfig(AppConfig):
    ENV = 'production'
    DEBUG = False
    @classmethod
    def init_app(cls, app):
        AppConfig.init_app(app)
        assert os.environ.get('SECRET_KEY'), 'SECRET KEY IS NOT SET!'
        print('Production Environment Activated.')


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
