#!/bin/bash
sed '/^0/d' $1 | sed '/^[0-9]\{1,16\} 0/d'