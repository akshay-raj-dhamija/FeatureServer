import logging
from flask import Flask
from flask import request


def api_interface(cache_obj, logger):
    logger.info("Starting flask_processing")

    app = Flask(__name__)
    # Suppressing flask logs only to error because we want the input to be accepted by main process
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    @app.route("/data")
    def get_oldest_entry():
        return cache_obj.get()

    @app.route("/after")
    def get_entries_after():
        args = request.args
        return cache_obj.get_after(float(args["timestamp"]))

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()
        logger.info("Server shutting down...")
        return "Server shutting down..."

    app.run(host="0.0.0.0", port="9999", debug=False)
    return
