BINS	=	my_torch_analyzer my_torch_generator

all:	$(BINS)

$(BINS):	%:	src/bin/%
	cp $< $@
	chmod +x $@

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

fclean:	clean
	rm -f $(BINS)

re:	fclean all

.PHONY:	all clean fclean re
