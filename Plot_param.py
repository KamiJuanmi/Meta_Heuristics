import os
import xlsxwriter
import glob
#Script that moves all the parameters to an excell in other to make graphs

#In order to this file to work, main.py and algorithms.py need to have the prints modified
def move_to_excell(workbook):
        row_index = 2

        col = 65
        aux = 0

        worksheet = workbook.add_worksheet()
        worksheet.write('B1', 'reinsert_best_position')
        worksheet.write('C1', 'dummy_back_exchange')
        worksheet.write('D1', 'random_removal')
        worksheet.write('E1', 'get_car_full')
        worksheet.write('F1', 'one_reinsert_best_car')
        with open('plot.txt') as fp:
                line = fp.readline()
                cnt = 0
                while line:
                        worksheet.write_number(chr(col) + str(row_index), float(line))
                        line = fp.readline()
                        cnt += 1
                        col += 1
                        if(cnt % 5 == 0):
                                col = 66
                                worksheet.write_number('A' + str(row_index), row_index-1)
                                row_index += 1

data_sets = glob.glob("../data/*txt")

workbook = xlsxwriter.Workbook("Plot.xlsx")

for i in data_sets:
        os.system("python main.py " + i + " 2 >> plot.txt")
        move_to_excell(workbook)
        os.system('rm plot.txt')

workbook.close()

