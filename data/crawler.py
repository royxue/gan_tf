from icrawler.builtin import GoogleImageCrawler


def main():
    google_crawler = GoogleImageCrawler(parser_threads=4, downloader_threads=8,
                                        storage={'root_dir': './kojiharu'})
    google_crawler.crawl(keyword='kojiharu', max_num=1000,
                         date_min=None, date_max=None,
                         min_size=(200,200), max_size=None)

if __name__ == '__main__':
    main()