研究一下 threads 的功能
例如 當使用者貼出 `https://www.threads.com/@myun.60761/post/DVnP0ATET7d?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw` 這種網址
自動將圖片或影片全部下載下來 文字也記錄下來
注意 `?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw` 這段參數是可以忽略的

後續我會想整合進去 discord bot 裡面, 但目前我覺得可以先專注於完成這個 `threads.py` 的功能
我想確定他是否能運作 後續再決定是否要整合進去 discord bot

目前只需要完成 `threads.py`
基本上就是要嚴格遵守 type hint 等等規範, 完成一個function

這個 function 會將貼文文字印出 並將 圖片 或 影片下載下來

我完成了一個新功能, 他會獲取 threads 貼文中的純文字與影片和圖片
我寫在 `src/discordbot/utils/threads.py`
請幫我把這個功能整合進去 discord bot 的 `src/discordbot/cogs`

具體作法就是標記原始的使用者, 然後將這些文字和圖片當成訊息傳出 (如果圖片或影片不存在就單純傳文字就行)
但要注意一個地方 假設影片或圖片過大, 可能就不能傳出文字和圖片 應該可以透過網址的方式傳送
我有在 ThreadsOutput 裡面有留下 media_urls 的欄位可以使用, 我覺得可以先檢查檔案大小 如果超過 25mb 就將網址放在最後面傳送
反之低於 25mb 就直接傳送圖片或影片檔案

discordbot 對於檔案大小的限制是 單次訊息總和 25mb, 單次最大文件數是 10 (影片 + 圖片總和不能超過 10 個) 這些限制都要注意到
