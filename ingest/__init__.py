"""Local ingestion pipeline for the portfolio chatbot.

Walks the user's code directory, builds sanitized project "cards", and emits a
deploy-safe bundle. Raw code never leaves the machine.
"""
