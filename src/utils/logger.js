// src/utils/logger.js
// 轻量分级日志。生产可经 env 控级别，零依赖。
const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
const COLOR = {
  debug: '\x1b[90m', info: '\x1b[36m', warn: '\x1b[33m', error: '\x1b[31m', reset: '\x1b[0m',
};

export class Logger {
  constructor(name = 'stellar', level = (process.env.STELLAR_LOG || 'info')) {
    this.name = name;
    this.level = LEVELS[level] ?? LEVELS.info;
  }
  _log(level, msg, ...rest) {
    if (LEVELS[level] < this.level) return;
    const c = COLOR[level] || '';
    const ts = new Date().toISOString().slice(11, 23);
    console.log(`${c}[${ts}][${level.toUpperCase()}][${this.name}]${COLOR.reset} ${msg}`, ...rest);
  }
  debug(...a) { this._log('debug', ...a); }
  info(...a) { this._log('info', ...a); }
  warn(...a) { this._log('warn', ...a); }
  error(...a) { this._log('error', ...a); }
}

export const log = new Logger();
