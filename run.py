from application import create_app

# create a copy of the application
app = create_app()

# and run it
if __name__ == '__main__':
    app.run()
