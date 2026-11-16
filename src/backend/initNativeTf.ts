import * as path from 'path';

// Node.js v24 removed deprecated `util.isNullOrUndefined`, which @tensorflow/tfjs-node depends on.
// Polyfill it so tfjs-node works flawlessly on Node v24:
const util = require('util');
if (!util.isNullOrUndefined) {
  util.isNullOrUndefined = (val: any) => val === undefined || val === null;
}

// This must run before @tensorflow/tfjs-node is required or imported
const tfDllPath = path.resolve(process.cwd(), 'node_modules/@tensorflow/tfjs-node/deps/lib');
process.env.PATH = `${tfDllPath};${process.env.PATH || ''}`;

// Set threadpool sizes for all available logical cores
const os = require('os');
const numCores = os.cpus().length || 4;
process.env.UV_THREADPOOL_SIZE = String(numCores);
process.env.OMP_NUM_THREADS = String(numCores);
process.env.TF_NUM_INTEROP_THREADS = String(numCores);
process.env.TF_NUM_INTRAOP_THREADS = String(numCores);

// Now load the native bindings
require('@tensorflow/tfjs-node');

