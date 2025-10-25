from fastapi import FastAPI,File,UploadFile
from fastapi.responses import JSONResponse
from sudoku import extract_and_solve
import numpy as np
import cv2
app=FastAPI()

@app.post("/solve")
async def solve(file:UploadFile=File(...)):
    try:

        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        solved_board = extract_and_solve(img)
        solved_board=np.array(solved_board).astype(int).tolist()

        return JSONResponse(content={
            "status": "success",
            "solved_board": solved_board
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        })