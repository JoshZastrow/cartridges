import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import cartridges

    return (cartridges,)


if __name__ == "__main__":
    app.run()
