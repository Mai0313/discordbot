研究一下 threads 的功能
例如 當使用者貼出 `https://www.threads.com/@myun.60761/post/DVnP0ATET7d?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw` 這種網址
自動將圖片或影片全部下載下來 文字也記錄下來
注意 `?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw` 這段參數是可以忽略的

後續我會想整合進去 discord bot 裡面, 但目前我覺得可以先專注於完成這個 `threads.py` 的功能
我想確定他是否能運作 後續再決定是否要整合進去 discord bot

目前只需要完成 `threads.py`
基本上就是要嚴格遵守 type hint 等等規範, 完成一個function

這個 function 會將貼文文字印出 並將 圖片 或 影片下載下來

以上功能已完成 位置在 `src/discordbot/utils/threads.py`
請幫我整合進去 discord bot 的 `src/discordbot/cogs`

具體作法就是標記原始的使用者, 然後將這些文字和圖片當成訊息傳出 (圖片和影片如果不存在 可以跳過 單純傳文字就行)
但要注意一個地方 假設影片或圖片過大, 可能就不能傳出文字和圖片 應該可以透過網址的方式傳送
我不確定用 embedded message 是否能完美適配網址的情況 但我在 ThreadsOutput 裡面有留下 media_urls 的欄位可以使用
我也不確定是否其實不用下載下來 直接傳網址可能壓力更小一點 你幫我想一下有沒有好辦法
