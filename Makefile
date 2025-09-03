# Top-level Makefile for spydcmtk project
#

.PHONY: help docs docs-clean docs-serve

help:
	@echo "Available targets:"
	@echo "  docs        - Build documentation"
	@echo "  docs-clean  - Clean documentation build"
	@echo "  docs-serve  - Serve documentation locally"
	@echo "  help        - Show this help message"

docs:
	@echo "Building documentation..."
	cd sphinx && make html

docs-clean:
	@echo "Cleaning documentation build..."
	cd sphinx && make clean

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd sphinx && python -m http.server 8000 --directory _build/html
