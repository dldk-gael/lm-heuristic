from grammar_backend import app
from grammar_backend.views import views

app.register_blueprint(views)

if __name__ == "__main__":
    app.run()