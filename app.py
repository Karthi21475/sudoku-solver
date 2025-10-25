import os
import cv2
import numpy as np
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sudoku import extract_and_solve
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app=FastAPI()

fe_url = os.getenv("FRONTEND_URL", "*")

origins = [fe_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Sudoku solver API is running!"}

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
            "solved_board":solved_board
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        })
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)