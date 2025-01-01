import { useEffect, useState, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowDown, ArrowLeft, ArrowRight, RotateCw } from 'lucide-react'

// Tetromino shapes
const TETROMINOES = {
  I: [[1, 1, 1, 1]],
  O: [[1, 1], [1, 1]],
  T: [[0, 1, 0], [1, 1, 1]],
  L: [[1, 0], [1, 0], [1, 1]],
  J: [[0, 1], [0, 1], [1, 1]],
  S: [[0, 1, 1], [1, 1, 0]],
  Z: [[1, 1, 0], [0, 1, 1]]
}

type Position = { x: number; y: number }
type Piece = { shape: number[][]; pos: Position }

function App() {
  const BOARD_WIDTH = 10
  const BOARD_HEIGHT = 20
  const [board, setBoard] = useState<number[][]>(Array(BOARD_HEIGHT).fill(0).map(() => Array(BOARD_WIDTH).fill(0)))
  const [currentPiece, setCurrentPiece] = useState<Piece | null>(null)
  const [gameOver, setGameOver] = useState(false)
  const [score, setScore] = useState(0)

  // Create new piece
  const createNewPiece = useCallback(() => {
    const shapes = Object.values(TETROMINOES)
    const shape = shapes[Math.floor(Math.random() * shapes.length)]
    return {
      shape,
      pos: { x: Math.floor((BOARD_WIDTH - shape[0].length) / 2), y: 0 }
    }
  }, [])

  // Check collision
  const hasCollision = (piece: Piece, board: number[][]) => {
    return piece.shape.some((row, dy) =>
      row.some((value, dx) => {
        const newY = piece.pos.y + dy
        const newX = piece.pos.x + dx
        return (
          value !== 0 &&
          (newY >= BOARD_HEIGHT ||
            newX < 0 ||
            newX >= BOARD_WIDTH ||
            (newY >= 0 && board[newY][newX] !== 0))
        )
      })
    )
  }

  // Merge piece with board
  const mergePieceWithBoard = (piece: Piece, board: number[][]) => {
    const newBoard = board.map(row => [...row])
    piece.shape.forEach((row, y) => {
      row.forEach((value, x) => {
        if (value !== 0) {
          const newY = piece.pos.y + y
          const newX = piece.pos.x + x
          if (newY >= 0) {
            newBoard[newY][newX] = value
          }
        }
      })
    })
    return newBoard
  }

  // Move piece
  const movePiece = (dx: number, dy: number) => {
    if (!currentPiece || gameOver) return

    const newPiece = {
      ...currentPiece,
      pos: { x: currentPiece.pos.x + dx, y: currentPiece.pos.y + dy }
    }

    if (!hasCollision(newPiece, board)) {
      setCurrentPiece(newPiece)
    } else if (dy > 0) {
      // Piece has landed
      const newBoard = mergePieceWithBoard(currentPiece, board)
      setBoard(newBoard)
      
      // Check for completed lines
      const completedLines = newBoard.reduce((acc, row, i) => {
        if (row.every(cell => cell !== 0)) acc.push(i)
        return acc
      }, [] as number[])

      if (completedLines.length > 0) {
        const newScore = score + (completedLines.length * 100)
        setScore(newScore)
        
        // Remove completed lines
        const filteredBoard = newBoard.filter((_, i) => !completedLines.includes(i))
        const newLines = Array(completedLines.length).fill(0).map(() => Array(BOARD_WIDTH).fill(0))
        setBoard([...newLines, ...filteredBoard])
      }

      // Create new piece
      const newPiece = createNewPiece()
      if (hasCollision(newPiece, newBoard)) {
        setGameOver(true)
      } else {
        setCurrentPiece(newPiece)
      }
    }
  }

  // Rotate piece
  const rotatePiece = () => {
    if (!currentPiece || gameOver) return

    const rotated = currentPiece.shape[0].map((_, i) =>
      currentPiece.shape.map(row => row[i]).reverse()
    )

    const newPiece = {
      ...currentPiece,
      shape: rotated
    }

    if (!hasCollision(newPiece, board)) {
      setCurrentPiece(newPiece)
    }
  }

  // Handle keyboard controls
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          movePiece(-1, 0)
          break
        case 'ArrowRight':
          movePiece(1, 0)
          break
        case 'ArrowDown':
          movePiece(0, 1)
          break
        case 'ArrowUp':
          rotatePiece()
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [currentPiece, board])

  // Start game and piece drop interval
  useEffect(() => {
    if (!currentPiece && !gameOver) {
      setCurrentPiece(createNewPiece())
    }

    const dropInterval = setInterval(() => {
      if (!gameOver) {
        movePiece(0, 1)
      }
    }, 1000)

    return () => clearInterval(dropInterval)
  }, [currentPiece, gameOver])

  // Reset game
  const resetGame = () => {
    setBoard(Array(BOARD_HEIGHT).fill(0).map(() => Array(BOARD_WIDTH).fill(0)))
    setCurrentPiece(null)
    setGameOver(false)
    setScore(0)
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <Card className="w-96">
        <CardHeader>
          <CardTitle className="text-center">Russian Square Game</CardTitle>
          <div className="text-sm text-center text-gray-500">
            Use arrow keys to play:<br/>
            ← → : Move left/right<br/>
            ↓ : Move down faster<br/>
            ↑ : Rotate piece
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 text-center font-semibold text-lg">Score: {score}</div>
          
          <div className="bg-white border-2 border-gray-200 p-1">
            {board.map((row, y) => (
              <div key={y} className="flex">
                {row.map((cell, x) => {
                  let isCurrent = false
                  if (currentPiece) {
                    const pieceY = y - currentPiece.pos.y
                    const pieceX = x - currentPiece.pos.x
                    if (
                      pieceY >= 0 &&
                      pieceY < currentPiece.shape.length &&
                      pieceX >= 0 &&
                      pieceX < currentPiece.shape[0].length
                    ) {
                      isCurrent = currentPiece.shape[pieceY][pieceX] !== 0
                    }
                  }
                  const colors = ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-pink-500']
                  const colorIndex = Math.abs(y + x) % colors.length
                  return (
                    <div
                      key={x}
                      className={`w-6 h-6 border border-gray-300
                        ${cell ? colors[colorIndex] : isCurrent ? 'bg-blue-500' : 'bg-gray-50'}`}
                    />
                  )
                })}
              </div>
            ))}
          </div>

          {gameOver && (
            <div className="mt-4 text-center">
              <div className="text-red-500 mb-2">Game Over!</div>
              <Button onClick={resetGame}>Play Again</Button>
            </div>
          )}

          <div className="mt-4 flex justify-center gap-2">
            <Button size="icon" onClick={() => movePiece(-1, 0)}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <Button size="icon" onClick={() => movePiece(0, 1)}>
              <ArrowDown className="h-4 w-4" />
            </Button>
            <Button size="icon" onClick={() => movePiece(1, 0)}>
              <ArrowRight className="h-4 w-4" />
            </Button>
            <Button size="icon" onClick={rotatePiece}>
              <RotateCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App
