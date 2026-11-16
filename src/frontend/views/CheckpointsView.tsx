import React, { useState } from 'react';
import { useStore } from '../lib/store';
import { api } from '../lib/api';
import {
  Panel,
  Button,
  Badge,
  Note,
  Table,
  Th,
  Td,
  EmptyState,
  Modal,
  Stat,
} from '../components/ui';
import * as fmt from '../lib/format';
import { CheckpointMetadata } from '../../shared/types';

/**
 * ============================================================================
 * Checkpoints - saved weights, and resuming from them
 * ============================================================================
 *
 * Loading a checkpoint restores three things, and the distinction matters:
 *
 *   1. The WEIGHTS.
 *   2. The STEP COUNT, so the learning-rate schedule picks up where it was.
 *   3. ADAM'S MOMENT ESTIMATES, where present.
 *
 * Without (3) a "resumed" run restarts the optimizer from zero momentum with
 * no learned per-parameter step sizes, which produces a visible loss bump and
 * wastes the next few dozen steps re-deriving what it already knew. The table
 * therefore shows explicitly whether each checkpoint can truly resume.
 *
 * Each checkpoint also carries its own architecture and vocabulary, so loading
 * one adopts its configuration rather than failing on a shape mismatch - and a
 * checkpoint trained on a different corpus is refused outright, because token
 * IDs would mean different characters and the output would be fluent nonsense.
 */

export const CheckpointsView: React.FC = () => {
  const { checkpoints, refreshCheckpoints, status, reportError, pushToast, step } = useStore();

  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<CheckpointMetadata | null>(null);

  const isRunning = status === 'running';

  const load = async (checkpoint: CheckpointMetadata) => {
    setLoadingId(checkpoint.id);
    try {
      const result = await api.loadCheckpoint(checkpoint.id);
      pushToast('success', result.message, 'Architecture and optimizer state were restored too.');
      // The loaded checkpoint changes step, config and parameter count, so the
      // whole store needs re-syncing rather than a partial patch.
      window.setTimeout(() => window.location.reload(), 600);
    } catch (error) {
      reportError(error, 'Could not load checkpoint');
    } finally {
      setLoadingId(null);
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await api.deleteCheckpoint(pendingDelete.id);
      await refreshCheckpoints();
      pushToast('info', `Deleted ${pendingDelete.name}`);
    } catch (error) {
      reportError(error, 'Could not delete checkpoint');
    } finally {
      setPendingDelete(null);
    }
  };

  const best = checkpoints.find((c) => c.isBest);
  const totalBytes = checkpoints.reduce((acc, c) => acc + c.fileSizeBytes, 0);

  return (
    <div className="flex flex-col gap-3">
      {checkpoints.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Checkpoints" value={checkpoints.length} />
          <Stat
            label="Best val loss"
            value={fmt.loss(best?.valLoss)}
            tone={best ? 'success' : 'default'}
            hint={best ? `step ${fmt.int(best.step)}` : undefined}
          />
          <Stat label="Disk used" value={fmt.bytes(totalBytes)} />
          <Stat label="Current step" value={fmt.int(step)} hint="model in memory" />
        </div>
      )}

      <Panel
        title="Saved checkpoints"
        description="Binary .mgpt files in ./checkpoints, each carrying its own config and vocabulary."
        actions={
          <Button size="sm" variant="ghost" onClick={refreshCheckpoints}>
            Refresh
          </Button>
        }
        flush
      >
        {checkpoints.length === 0 ? (
          <div className="p-3.5">
            <EmptyState
              title="No checkpoints yet"
              description="One is written whenever validation loss improves, plus periodically according to the checkpoint interval."
            />
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Name</Th>
                <Th align="right">Step</Th>
                <Th align="right">Train</Th>
                <Th align="right">Val</Th>
                <Th align="right">Params</Th>
                <Th>Architecture</Th>
                <Th align="right">Size</Th>
                <Th>Saved</Th>
                <Th align="right">Resume</Th>
                <Th align="right" />
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((checkpoint) => (
                <tr key={checkpoint.id} className="hover:bg-surface-hover">
                  <Td className="font-medium text-content-primary">
                    <span className="flex items-center gap-2">
                      {checkpoint.name}
                      {checkpoint.isBest && <Badge tone="success">best</Badge>}
                    </span>
                  </Td>
                  <Td align="right" mono>
                    {fmt.int(checkpoint.step)}
                  </Td>
                  <Td align="right" mono>
                    {fmt.loss(checkpoint.trainLoss)}
                  </Td>
                  <Td align="right" mono className="text-content-primary">
                    {fmt.loss(checkpoint.valLoss)}
                  </Td>
                  <Td align="right" mono>
                    {fmt.compactInt(
                      checkpoint.parameters ??
                        estimateParameters(checkpoint)
                    )}
                  </Td>
                  <Td mono title={JSON.stringify(checkpoint.modelConfig, null, 2)}>
                    {checkpoint.modelConfig.numLayers}L·{checkpoint.modelConfig.numHeads}H·d
                    {checkpoint.modelConfig.embeddingDim}
                  </Td>
                  <Td align="right" mono>
                    {fmt.bytes(checkpoint.fileSizeBytes)}
                  </Td>
                  <Td mono>{fmt.dateTime(checkpoint.createdAt)}</Td>
                  <Td align="right">
                    {checkpoint.hasOptimizerState ? (
                      <Badge
                        tone="success"
                        title="Adam moment estimates are stored, so training resumes with momentum intact."
                      >
                        full
                      </Badge>
                    ) : (
                      <Badge
                        tone="neutral"
                        title="Weights only. Resuming restarts Adam from zero momentum, which causes a small loss bump."
                      >
                        weights
                      </Badge>
                    )}
                  </Td>
                  <Td align="right">
                    <span className="flex justify-end gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        loading={loadingId === checkpoint.id}
                        disabled={isRunning}
                        onClick={() => load(checkpoint)}
                        title="Restore weights, step count and optimizer state"
                      >
                        Load
                      </Button>
                      <a
                        href={api.checkpointDownloadUrl(checkpoint.id)}
                        download
                        className="inline-flex h-7 items-center rounded-md border border-transparent px-2 text-xs font-medium text-content-secondary hover:bg-surface-hover hover:text-content-primary"
                      >
                        Download
                      </a>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={isRunning}
                        onClick={() => setPendingDelete(checkpoint)}
                        className="text-danger hover:bg-danger-subtle"
                      >
                        Delete
                      </Button>
                    </span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Panel>

      {isRunning && (
        <Note tone="info">
          Checkpoints cannot be loaded or deleted while a run is in progress. Pause or stop it
          first.
        </Note>
      )}

      <Panel title="About the .mgpt format">
        <div className="grid grid-cols-1 gap-3 text-xs leading-relaxed text-content-secondary sm:grid-cols-2">
          <div>
            <p className="mb-1.5 font-semibold text-content-primary">What is inside</p>
            <p>
              A four-byte magic marker, a JSON header describing the architecture, vocabulary, step
              and metrics, then one contiguous block of float32 weights. The header can be read
              without touching the weights, which is how this table lists architecture and loss for
              every file cheaply.
            </p>
          </div>
          <div>
            <p className="mb-1.5 font-semibold text-content-primary">Why not JSON</p>
            <p>
              A float32 occupies 4 bytes in memory but 8 to 20 ASCII bytes written as a decimal
              string. Storing weights as JSON numbers made checkpoints several times larger than
              the data they held, and parsing a multi-million-element number array allocated a
              boxed value per weight.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  );
};

/**
 * Derives a parameter count from a checkpoint's architecture, for older files
 * whose header predates the stored count.
 */
function estimateParameters(checkpoint: CheckpointMetadata): number {
  const m = checkpoint.modelConfig;
  const D = m.embeddingDim;
  const V = m.vocabularySize;
  const hidden = D * (m.feedForwardMultiplier || 4);
  const perLayer = 4 * (D * D + D) + (D * hidden + hidden + hidden * D + D) + 4 * D;
  const tied = m.tieWeights !== false;
  return V * D + m.contextLength * D + perLayer * m.numLayers + 2 * D + (tied ? V : D * V + V);
}
