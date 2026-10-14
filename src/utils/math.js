// src/utils/math.js
// 向量与数值基元 —— 检索增强 / 相似度 / 注意力机制的数学底座。
// 零重写：仅实现不可外包的轻量基元；重计算交给 transformers.js / hnswlib。

export function dot(a, b) {
  if (a.length !== b.length) throw new Error(`维度不匹配: ${a.length} vs ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function cosineSimilarity(a, b) {
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

export function normalize(a) {
  const n = norm(a);
  return n === 0 ? a.map(() => 0) : a.map((x) => x / n);
}

export function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

export function softmax(arr, t = 1.0) {
  const m = Math.max(...arr);
  const exps = arr.map((x) => Math.exp((x - m) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// topK：返回按分数降序的 [{index, score}]，稳定且只扫描一次。
export function topK(arr, k) {
  const idx = arr.map((score, index) => ({ index, score }));
  idx.sort((a, b) => b.score - a.score);
  return idx.slice(0, k);
}

export function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
}

export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}
