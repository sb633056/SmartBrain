'use client'
import React, { useState, useRef } from 'react'
import { runSmartBrain } from '@/lib/api'

export default function ChatConsole() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<Array<{ q: string; a: string }>>([])

  async function handleRun() {
    if (!prompt.trim()) return
    setLoading(true)

    try {
      const res = await runSmartBrain(prompt)
      setOutput(res)
      setHistory((h) => [{ q: prompt, a: res }, ...h])
      setPrompt('')
    } catch (e: any) {
      setOutput(`Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:ring focus:ring-indigo-200 p-2"
          placeholder="Ask SmartBrain anything..."
        />
      </div>

      <div className="flex gap-2">
        <button
          onClick={handleRun}
          disabled={loading}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg disabled:opacity-60"
        >
          {loading ? 'Thinking…' : 'Run'}
        </button>

        <button
          onClick={() => {
            setPrompt('')
            setOutput('')
          }}
          className="px-4 py-2 border rounded-lg"
        >
          Clear
        </button>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Output</label>
        <div className="mt-1 rounded-md border p-3 bg-gray-50 whitespace-pre-wrap min-h-[120px]">
          {output}
        </div>
      </div>
    </div>
  )
}
