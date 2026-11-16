export interface PredefinedDataset {
  name: string;
  url: string;
  description: string;
  sizeMb: string;
}

export const PREDEFINED_DATASETS: PredefinedDataset[] = [
  {
    name: 'tinyshakespeare',
    url: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    description: 'Shakespeare plays — quick baseline benchmark',
    sizeMb: '~1.1 MB',
  },
  {
    name: 'sherlock-holmes',
    url: 'https://www.gutenberg.org/files/1661/1661-0.txt',
    description: 'The Adventures of Sherlock Holmes by Arthur Conan Doyle',
    sizeMb: '~0.6 MB',
  },
  {
    name: 'pride-and-prejudice',
    url: 'https://www.gutenberg.org/files/1342/1342-0.txt',
    description: 'Pride and Prejudice by Jane Austen — rich 19th-century English',
    sizeMb: '~0.8 MB',
  },
  {
    name: 'frankenstein',
    url: 'https://www.gutenberg.org/files/84/84-0.txt',
    description: 'Frankenstein by Mary Shelley — gothic sci-fi narrative',
    sizeMb: '~0.5 MB',
  },
  {
    name: 'dracula',
    url: 'https://www.gutenberg.org/files/345/345-0.txt',
    description: 'Dracula by Bram Stoker — epistolary horror classic',
    sizeMb: '~0.9 MB',
  },
  {
    name: 'moby-dick',
    url: 'https://www.gutenberg.org/files/2701/2701-0.txt',
    description: 'Moby Dick by Herman Melville — comprehensive vocabulary',
    sizeMb: '~1.3 MB',
  },
  {
    name: 'war-and-peace',
    url: 'https://www.gutenberg.org/files/2600/2600-0.txt',
    description: 'War and Peace by Leo Tolstoy — substantial text corpus',
    sizeMb: '~3.3 MB',
  },
  {
    name: 'count-of-monte-cristo',
    url: 'https://www.gutenberg.org/files/1184/1184-0.txt',
    description: 'The Count of Monte Cristo by Alexandre Dumas — epic adventure',
    sizeMb: '~2.7 MB',
  },
  {
    name: 'les-miserables',
    url: 'https://www.gutenberg.org/files/135/135-0.txt',
    description: 'Les Misérables by Victor Hugo — massive literary work',
    sizeMb: '~3.4 MB',
  },
];
