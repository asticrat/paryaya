class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunks = [];
    this._len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    this._chunks.push(ch.slice());
    this._len += ch.length;
    if (this._len >= 4096) {
      const buf = new Float32Array(this._len);
      let off = 0;
      for (const c of this._chunks) { buf.set(c, off); off += c.length; }
      this.port.postMessage(buf.buffer, [buf.buffer]);
      this._chunks = [];
      this._len = 0;
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
