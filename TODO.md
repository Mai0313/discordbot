- 處理 Reasoning Model 的思考過程, 將思考過程顯示在機器人回應中
- 續寫/重生成功能: 透過點擊重新生成的按鈕, 利用 `previous_response_id ` 快速重新生成圖片與文字, 限定提問者才能點該按鈕
- 用量/延遲小註記: 顯示回覆耗時、token 估算（若 API 回傳 usage）並透過 `Embed` 顯示

src/discordbot/cogs/auction.py 這部分的代碼實在是太長了
我覺得應該把功能切開放到不同的文件中

請幫我將功能切開以後 放到 `src/discordbot/cogs/_auction` 資料夾中
不要做任何修改 單純將功能分門別類歸類到不同檔案即可
主要的 `setup` 和 `AuctionCogs` 依然放在 `src/discordbot/cogs/auction.py` 但其他東西幫我分類放進 `_auction`
