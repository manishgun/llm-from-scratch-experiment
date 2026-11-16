/**
 * Vitest global setup.
 *
 * Loads the native TensorFlow backend before any test imports tfjs. Without
 * this the whole suite runs on the pure-JavaScript CPU backend, which makes
 * the gradient check and overfit tests roughly ten times slower - slow enough
 * that people stop running them.
 */
import '../src/backend/initNativeTf';
