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
請幫我整合進去 discord bot 的 src/discordbot/cogs
