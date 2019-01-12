from bs4 import BeautifulSoup as Soup
import pickle
import re
from collections import Counter
import numpy as np
from random import shuffle
import asyncio
import aiohttp
import async_timeout


def make_links():
    """make some links to Court of Appeal cases on bailii.org"""
    years = [y for y in range(1980, 2018)]
    nums = list(range(30))
    combos = [(y, n) for y in years for n in nums]
    base_addr = "http://www.bailii.org/ew/cases/EWCA/Civ/"
    return [''.join([base_addr, str(y), "/", str(n), ".html"]) for (y, n) in combos]


def get_batches(urls, n):
    """group the urls into batches of max length n - currently unused"""
    print("making batches of max length {} from {} urls".format(n, len(urls)))
    return [urls[i:len(urls):n] for i in range(n)]


async def get_text(session, url):
    """download case from url, return BeautifulSoup text"""
    with async_timeout.timeout(1200):
        try:
            async with session.get(url) as response:
                t = await response.text()
                return url, Soup(t, "lxml").text
        except Exception as e:
            print(e)


async def coordinate_downloads(urls):
    """assemble the download tasks"""
    tasks = []
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        print("retrieving {} cases".format(len(urls)))
        for url in urls:
            tasks.append(asyncio.ensure_future(get_text(session, url)))
        return await asyncio.gather(*tasks)


def find_relevant_parts(raw, pattern):
    """extract passages matching pattern"""
    return re.findall(pattern, str(raw))


def build_vocab(text):
    """takes a list of strings and returns a list of words and a counter of each word frequency"""
    n = Counter([word.lower() for para in text for word in para.split()])
    vocab = [x[0] for x in sorted(n.items(), key=lambda x: -x[1])]
    return vocab, n


def vocab_index(vocab):
    """takes a list of words and returns mappings from word to index and index to work"""
    idx2w = dict(enumerate(vocab))
    return idx2w, {w: idx for (idx, w) in idx2w.items()}


def encode(text, w2idx):
    """takes a list of (url, string) tuples and returns an encoding based on w2idx"""
    return text[0], np.array([w2idx[word.lower()] for para in text[1] for word in para.split()])


def decode(encoding, idx2w):
    return [idx2w[n] for n in encoding[1]]


def binarize(encoded, n):
    """takes the numerically encoded text and the number of unique words and returns a sparse matrix"""
    return [(enc[0], np.array([1 if i in enc[1] else 0 for i in range(n)])) for enc in encoded]


def main():
    """download cases, build vocabulary model"""

    """simple regex to find legal discussion, based on citation of authorities in [****] ____ ___ format"""
    pattern = re.compile(".*\[\d\d\d\d\].*")

    """get urls for cases"""
    urls = make_links()
    shuffle(urls)

    """async downloads"""
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(coordinate_downloads(urls))

    """retrieve relevant extracts"""
    print("retrieving relevant extracts...")
    extracts = [(url, find_relevant_parts(text, pattern)) for (url, text) in loop.run_until_complete(future)]

    """build vocabulary and mapping"""
    print("counting words...")
    sentences = [sentence for x in extracts for sentence in x[1]]
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    print("found {} total words".format(len(words)))
    print("building vocabulary...")
    vocab, vocab_count = build_vocab(sentences)
    idx2w, w2idx = vocab_index(vocab)
    print("{} unique words".format(len(vocab)))

    """encode cases"""
    print("encoding...")
    encoded_cases = [encode(extract, w2idx) for extract in extracts if len(extract[1]) > 1]
    binarized = binarize(encoded_cases, len(vocab))

    """save encoded cases and models"""
    print("saving...")
    with open("encoded_cases.pkl", "wb") as f:
        pickle.dump(binarized, f)
    with open("w2idx.pkl", "wb") as f:
        pickle.dump((w2idx, idx2w), f)


if __name__ == "__main__":
    main()
