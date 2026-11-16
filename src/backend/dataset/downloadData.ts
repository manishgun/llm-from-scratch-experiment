import { DatasetManager } from './datasetManager';

/**
 * CLI utility to trigger download of the Tiny Shakespeare dataset.
 */
async function run() {
  console.log('Downloading Tiny Shakespeare dataset...');
  const manager = new DatasetManager();

  try {
    const file = await manager.ensureTinyShakespeare((pct) => {
      process.stdout.write(`\rDownloading: ${pct}%`);
    });
    console.log(`\nTiny Shakespeare downloaded successfully to: ${file}`);
  } catch (err) {
    console.error('\nError downloading dataset:', err);
    process.exit(1);
  }
}

if (require.main === module) {
  run();
}
