
import os 
import shutil
import tempfile 
from optparse import *




def orderbook_to_file(outfile, lines, count):
    outfile.writelines(lines[:count])
    
def fix(filename):
    infile = open(filename, 'r')
    outfile = tempfile.NamedTemporaryFile('w')
    # assume no individual orderbook is ever bigger than 200 
    curr_order_book = [None] * 200  
    curr_order_book_size = 0
    sep = "ORDERBOOK"
    sep_len = len(sep)
    skipped_lines = 0 
    dropped_books = 0 
    dropped_book_lines = 0 
    linecount = 0 
    for line in infile: 
        linecount += 1 
        if line[0:sep_len] == sep:
            orderbook_to_file(outfile, curr_order_book, curr_order_book_size)
            rest_of_line = line[sep_len:]
            if sep in rest_of_line:
                dropped_books += 1
                dropped_book_lines += curr_order_book_size 
                idx = rest_of_line.index(sep)
                curr_order_book[0] = rest_of_line[idx:]
            else:
                curr_order_book[0] = line
            curr_order_book_size = 1
                
            # TODO: handle case when two seps on one line 
        elif sep in line:
            dropped_books += 1 
            dropped_book_lines += curr_order_book_size 
            idx = line.index(sep)
            # drop whatever partial order book we have seen so far 
            curr_order_book[0] = line[idx:]
            curr_order_book_size = 1
        elif line == '\n' or line == '':
            skipped_lines += 1
        else: 
            curr_order_book[curr_order_book_size] = line
            curr_order_book_size += 1
            
    orderbook_to_file(outfile, curr_order_book, curr_order_book_size) 
    infile.close()
    outfile.flush()
    shutil.copy(outfile.name, filename) 
    # should delete the temp file 
    outfile.close() 
    print "Read", linecount, "lines" 
    print "Skipped", skipped_lines, "blank lines"
    print "Dropped", dropped_books, "order book(s) (", dropped_book_lines, "lines )"
     
if __name__ == '__main__':
    parser = OptionParser(usage = "usage: %prog path")
    (options, args) = parser.parse_args()
    if len(args) < 1: 
        print "filename not specified"
        exit(1) 
    else: 
        filename = args[0] 
        fix(filename) 
