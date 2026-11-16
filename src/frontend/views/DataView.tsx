import React, { useMemo, useRef, useState } from 'react';
import { useStore } from '../lib/store';
import { api, PredefinedDataset } from '../lib/api';
import {
  Panel,
  Button,
  Stat,
  Badge,
  Note,
  Segmented,
  Field,
  TextInput,
  TextArea,
  EmptyState,
  Table,
  Th,
  Td,
  Modal,
} from '../components/ui';
import { BarList } from '../components/charts';
import * as fmt from '../lib/format';

/**
 * ============================================================================
 * Data - the corpus, its splits, and how it is tokenized
 * ============================================================================
 *
 * The tokenizer switch is the most instructive control in the application,
 * because it exposes the tradeoff that governs every real language model:
 *
 *   character-level: tiny vocabulary, long sequences, must learn spelling
 *   word-level:      huge vocabulary, short sequences, spelling is free
 *
 * Switching it re-tokenizes the same text and the vocabulary size, sequence
 * length and resulting parameter count all move at once.
 */

export const DataView: React.FC = () => {
  const { dataset, refreshDataset, status, reportError, pushToast, config } = useStore();

  const [predefined, setPredefined] = useState<PredefinedDataset[] | null>(null);
  const [loadingName, setLoadingName] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadName, setUploadName] = useState('my-corpus');
  const [uploadText, setUploadText] = useState('');
  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isRunning = status === 'running';

  React.useEffect(() => {
    api.predefinedDatasets().then(setPredefined).catch(() => setPredefined([]));
  }, []);

  const loadPredefined = async (entry: PredefinedDataset) => {
    setLoadingName(entry.name);
    try {
      await api.loadDatasetUrl(entry.url, entry.name, dataset?.tokenizerType ?? 'char');
      await refreshDataset();
      pushToast('success', `Loaded ${entry.name}`, 'The model was rebuilt for the new vocabulary.');
    } catch (error) {
      reportError(error, `Could not load ${entry.name}`);
    } finally {
      setLoadingName(null);
    }
  };

  const switchTokenizer = async (type: 'char' | 'word') => {
    if (type === dataset?.tokenizerType) return;
    setBusy(true);
    try {
      await api.setTokenizer(type);
      await refreshDataset();
      pushToast(
        'success',
        `Switched to ${type}-level tokenization`,
        'The vocabulary changed, so the model was rebuilt.'
      );
    } catch (error) {
      reportError(error, 'Could not switch tokenizer');
    } finally {
      setBusy(false);
    }
  };

  const submitUpload = async () => {
    setBusy(true);
    try {
      await api.uploadDataset(uploadName, uploadText, dataset?.tokenizerType ?? 'char');
      await refreshDataset();
      setUploadOpen(false);
      setUploadText('');
      pushToast('success', 'Custom text loaded');
    } catch (error) {
      reportError(error, 'Upload failed');
    } finally {
      setBusy(false);
    }
  };

  const onPickFile = async (file: File) => {
    const text = await file.text();
    setUploadName(file.name.replace(/\.[^.]+$/, ''));
    setUploadText(text);
    setUploadOpen(true);
  };

  const loading = !dataset || dataset.name === 'loading' || dataset.totalCharacters === 0;

  // Sequence-length implications of the current tokenizer choice.
  const tokensPerWindow = config?.model.contextLength ?? 64;
  const charsCoveredEstimate = useMemo(() => {
    if (!dataset) return null;
    if (dataset.tokenizerType === 'char') return tokensPerWindow;
    // Rough average English word length plus its trailing space.
    return Math.round(tokensPerWindow * 5.7);
  }, [dataset, tokensPerWindow]);

  return (
    <div className="flex flex-col gap-3">
      {loading ? (
        <Panel title="Dataset">
          <EmptyState
            title="Preparing the dataset"
            description="Tiny Shakespeare is downloading on first run. This takes a few seconds."
          />
        </Panel>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="Corpus" value={dataset.name} hint={`${dataset.tokenizerType}-level`} />
            <Stat
              label="Characters"
              value={fmt.compactInt(dataset.totalCharacters)}
              hint={fmt.int(dataset.totalCharacters)}
            />
            <Stat
              label="Vocabulary"
              value={fmt.int(dataset.uniqueCharacters)}
              unit="tokens"
              hint={`baseline loss ln(V) = ${Math.log(dataset.uniqueCharacters).toFixed(2)}`}
              title="The number of distinct tokens. This sets the size of the embedding table and the output head, and the loss an untrained model scores."
            />
            <Stat
              label="Train split"
              value={fmt.compactInt(dataset.trainCharacters)}
              hint={fmt.percent(dataset.trainCharacters / dataset.totalCharacters, 0)}
            />
            <Stat
              label="Held out"
              value={fmt.compactInt(dataset.valCharacters + dataset.testCharacters)}
              hint="val + test, never trained on"
            />
          </div>

          <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,380px)]">
            <div className="flex min-w-0 flex-col gap-3">
              <Panel
                title="Tokenization"
                description="How raw text becomes the integers the model consumes."
                actions={
                  <Segmented
                    value={dataset.tokenizerType}
                    onChange={switchTokenizer}
                    disabled={isRunning || busy}
                    ariaLabel="Tokenizer"
                    options={[
                      { value: 'char', label: 'Character' },
                      { value: 'word', label: 'Word' },
                    ]}
                  />
                }
              >
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-md border border-edge-subtle bg-surface-input p-3">
                    <div className="label mb-1.5">Current choice</div>
                    <p className="text-xs leading-relaxed text-content-secondary">
                      {dataset.tokenizerType === 'char'
                        ? `Every character is one token. The vocabulary is just ${dataset.uniqueCharacters} entries, so the embedding table is tiny and the model can spell any word - but it must learn spelling from scratch, and a ${tokensPerWindow}-token window covers only ${tokensPerWindow} characters.`
                        : `Every whitespace-separated word is one token. The vocabulary is ${fmt.int(dataset.uniqueCharacters)} entries, so the embedding table and output head dominate the parameter count - but a ${tokensPerWindow}-token window covers roughly ${charsCoveredEstimate} characters of text.`}
                    </p>
                  </div>
                  <div className="rounded-md border border-edge-subtle bg-surface-input p-3">
                    <div className="label mb-1.5">The tradeoff</div>
                    <p className="text-xs leading-relaxed text-content-secondary">
                      Vocabulary size and sequence length pull in opposite directions. Real models
                      use subword tokenization (BPE) to sit between the two: a few tens of
                      thousands of tokens, no unknown words, and sequences several times shorter
                      than character level.
                    </p>
                  </div>
                </div>

                {isRunning && (
                  <Note tone="info" className="mt-3">
                    Pause the run to change tokenization - it changes the vocabulary, which requires
                    rebuilding the model.
                  </Note>
                )}
              </Panel>

              <Panel
                title="Text preview"
                description="The first part of the corpus, exactly as the tokenizer sees it."
              >
                <pre className="max-h-[260px] overflow-auto whitespace-pre-wrap break-words rounded-md border border-edge-subtle bg-surface-input p-3 font-mono text-xs leading-relaxed text-content-secondary">
                  {dataset.samplePreview}
                </pre>
              </Panel>

              <Panel
                title="Load a corpus"
                description="The model is rebuilt whenever the vocabulary changes."
                actions={
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".txt,text/plain"
                      className="hidden"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) void onPickFile(file);
                        event.target.value = '';
                      }}
                    />
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={isRunning}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Upload .txt
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={isRunning}
                      onClick={() => setUploadOpen(true)}
                    >
                      Paste text
                    </Button>
                  </>
                }
                flush
              >
                {predefined === null ? (
                  <div className="p-3.5 text-xs text-content-tertiary">Loading catalogue...</div>
                ) : (
                  <Table>
                    <thead>
                      <tr>
                        <Th>Corpus</Th>
                        <Th>Description</Th>
                        <Th align="right">Size</Th>
                        <Th align="right" />
                      </tr>
                    </thead>
                    <tbody>
                      {predefined.map((entry) => (
                        <tr key={entry.name} className="hover:bg-surface-hover">
                          <Td className="font-medium text-content-primary">
                            {entry.name}
                            {entry.name === dataset.name && (
                              <Badge tone="accent" className="ml-2">
                                loaded
                              </Badge>
                            )}
                          </Td>
                          <Td>{entry.description}</Td>
                          <Td align="right" mono>
                            {entry.sizeMb}
                          </Td>
                          <Td align="right">
                            <Button
                              size="sm"
                              variant="ghost"
                              loading={loadingName === entry.name}
                              disabled={isRunning || entry.name === dataset.name}
                              onClick={() => loadPredefined(entry)}
                            >
                              Load
                            </Button>
                          </Td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </Panel>
            </div>

            <div className="flex min-w-0 flex-col gap-3">
              <Panel
                title="Token frequency"
                description="The most common tokens in the corpus."
              >
                <BarList
                  items={dataset.charDistribution.slice(0, 22).map((entry) => ({
                    label: entry.char,
                    value: entry.count,
                    display: `${entry.percentage}%`,
                    mono: true,
                  }))}
                />
                <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
                  This distribution is what an untrained model has to beat. Simply always predicting
                  the most frequent token would already do better than uniform guessing - which is
                  roughly the first thing the model learns.
                </p>
              </Panel>

              <Panel title="Splits" description="Contiguous cuts, so no window spans two splits.">
                <div className="mb-2.5 flex h-5 overflow-hidden rounded-md border border-edge-subtle">
                  <div
                    className="flex items-center justify-center bg-accent text-2xs font-semibold text-white"
                    style={{
                      width: `${(dataset.trainCharacters / dataset.totalCharacters) * 100}%`,
                    }}
                    title={`train: ${fmt.int(dataset.trainCharacters)} tokens`}
                  >
                    train
                  </div>
                  <div
                    className="flex items-center justify-center text-2xs font-semibold text-white"
                    style={{
                      width: `${(dataset.valCharacters / dataset.totalCharacters) * 100}%`,
                      background: 'var(--series-val)',
                    }}
                    title={`validation: ${fmt.int(dataset.valCharacters)} tokens`}
                  >
                    val
                  </div>
                  <div
                    className="flex items-center justify-center bg-surface-hover text-2xs font-semibold text-content-secondary"
                    style={{
                      width: `${(dataset.testCharacters / dataset.totalCharacters) * 100}%`,
                    }}
                    title={`test: ${fmt.int(dataset.testCharacters)} tokens`}
                  >
                    test
                  </div>
                </div>
                <dl className="flex flex-col gap-1 text-xs">
                  {[
                    ['Train', dataset.trainCharacters],
                    ['Validation', dataset.valCharacters],
                    ['Test', dataset.testCharacters],
                  ].map(([label, value]) => (
                    <div key={label as string} className="flex justify-between gap-2">
                      <dt className="text-content-tertiary">{label}</dt>
                      <dd className="num text-content-primary">{fmt.int(value as number)}</dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
                  Validation loss is measured on a FIXED sample of windows from the validation
                  split, so two measurements differ only because the model changed - never because
                  the data did.
                </p>
              </Panel>
            </div>
          </div>
        </>
      )}

      <Modal
        open={uploadOpen}
        title="Load custom text"
        onClose={() => setUploadOpen(false)}
        width={560}
        footer={
          <>
            <Button variant="ghost" onClick={() => setUploadOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              loading={busy}
              disabled={uploadText.length < 32 || uploadName.length === 0}
              onClick={submitUpload}
            >
              Load
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <Field label="Name" help="Used for the saved file name. Letters, digits, dash and underscore.">
            {(id) => (
              <TextInput
                id={id}
                value={uploadName}
                onChange={(event) => setUploadName(event.target.value)}
              />
            )}
          </Field>
          <Field
            label="Text"
            valueDisplay={`${fmt.int(uploadText.length)} chars`}
            help="At least a few thousand characters is needed for the splits to be usable. More repetitive text learns faster."
          >
            {(id) => (
              <TextArea
                id={id}
                rows={10}
                value={uploadText}
                onChange={(event) => setUploadText(event.target.value)}
                placeholder="Paste text here..."
                className="font-mono"
              />
            )}
          </Field>
          {uploadText.length > 0 && uploadText.length < 5000 && (
            <Note tone="warning">
              This is quite short. With {fmt.int(uploadText.length)} characters, the validation
              split holds only about {fmt.int(Math.floor(uploadText.length * 0.1))} tokens, and the
              model will memorise the text almost immediately.
            </Note>
          )}
        </div>
      </Modal>
    </div>
  );
};
