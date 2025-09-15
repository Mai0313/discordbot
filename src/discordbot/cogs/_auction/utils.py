def get_currency_display(currency_type: str) -> str:
    """取得貨幣顯示文字"""
    currency_map = {"楓幣": "楓幣", "雪花": "雪花", "台幣": "台幣"}
    return currency_map.get(currency_type, "楓幣")
