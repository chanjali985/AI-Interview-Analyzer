import { useCallback, useEffect, useRef, useState } from 'react'

/** Pick a container the browser can actually record and the backend can decode. */
function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return ''
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ]
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || ''
}

export function extensionFor(mimeType = '') {
  if (mimeType.includes('ogg')) return 'ogg'
  if (mimeType.includes('mp4')) return 'mp4'
  return 'webm'
}

export const recordingSupported =
  typeof navigator !== 'undefined' &&
  Boolean(navigator.mediaDevices?.getUserMedia) &&
  typeof MediaRecorder !== 'undefined'

/**
 * Microphone recording with a live level meter and an optional hard time limit.
 */
export function useRecorder({ maxSeconds = 300, onStop } = {}) {
  const [state, setState] = useState('idle') // idle | requesting | recording | stopped | error
  const [seconds, setSeconds] = useState(0)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState('')

  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const rafRef = useRef(null)
  const audioContextRef = useRef(null)
  const secondsRef = useRef(0)
  const onStopRef = useRef(onStop)

  useEffect(() => {
    onStopRef.current = onStop
  }, [onStop])

  const cleanup = useCallback(() => {
    clearInterval(timerRef.current)
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close().catch(() => {})
    }
    audioContextRef.current = null
    setLevel(0)
  }, [])

  useEffect(() => cleanup, [cleanup])

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
    }
  }, [])

  const start = useCallback(async () => {
    if (!recordingSupported) {
      setError('This browser cannot record audio. Try the latest Chrome, Edge, Firefox or Safari.')
      setState('error')
      return
    }

    setError('')
    setState('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      streamRef.current = stream

      // Live input level, so the candidate can see the mic is working.
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext
        const context = new AudioContextClass()
        audioContextRef.current = context
        const source = context.createMediaStreamSource(stream)
        const analyser = context.createAnalyser()
        analyser.fftSize = 512
        source.connect(analyser)
        const buffer = new Uint8Array(analyser.frequencyBinCount)
        const tick = () => {
          analyser.getByteTimeDomainData(buffer)
          let sum = 0
          for (let i = 0; i < buffer.length; i += 1) {
            const value = (buffer[i] - 128) / 128
            sum += value * value
          }
          setLevel(Math.min(1, Math.sqrt(sum / buffer.length) * 3.2))
          rafRef.current = requestAnimationFrame(tick)
        }
        tick()
      } catch {
        /* the meter is a nicety; recording still works without it */
      }

      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        const type = recorder.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type })
        cleanup()
        setState('stopped')
        onStopRef.current?.(blob, { mimeType: type, seconds: secondsRef.current })
      }
      recorder.onerror = () => {
        cleanup()
        setError('Recording stopped unexpectedly. Please try again.')
        setState('error')
      }

      recorder.start(250)
      setSeconds(0)
      secondsRef.current = 0
      setState('recording')

      timerRef.current = setInterval(() => {
        secondsRef.current += 1
        setSeconds(secondsRef.current)
        if (secondsRef.current >= maxSeconds) stop()
      }, 1000)
    } catch (err) {
      cleanup()
      setState('error')
      setError(
        err?.name === 'NotAllowedError'
          ? 'Microphone access was blocked. Allow it in your browser settings and try again.'
          : err?.name === 'NotFoundError'
            ? 'No microphone was found. Plug one in and try again.'
            : 'Could not start recording: ' + (err?.message || 'unknown error'),
      )
    }
  }, [cleanup, maxSeconds, stop])

  const reset = useCallback(() => {
    cleanup()
    chunksRef.current = []
    secondsRef.current = 0
    setSeconds(0)
    setState('idle')
    setError('')
  }, [cleanup])

  return { state, seconds, level, error, start, stop, reset, isRecording: state === 'recording' }
}
