module prettytbl
#=  adapted from https://github.com/djsegal/Fussy.jl/blob/
 2cde77e2422dce590e770d2937a444c1eabfea7a/src/modules/HTMLElements.jl  =#
  using ArgCheck

  function _table(cur_matrix::Matrix; header_row=[], header_col=[], title=[],
      header=[], dp::Int=Inf)

    if isfinite(dp) && dp > 0
        cur_matrix = round.(cur_matrix*10^dp)/10^dp
    end

    cur_table = ""
    !isempty(header_col) && @argcheck size(cur_matrix, 1) == length(header_col)
    !isempty(header_row) && @argcheck size(cur_matrix, 2) == length(header_row)

    if !isempty(header)
      cur_table *= "<div><h3 style='padding: 10px'>$header</h2></div>"
    end

    cur_table *= "<table style='display: inline-block';>" #"<table style='float: $LR;'>"
    if !isempty(title)
        cur_table *= "<caption style='text-align: center'>$title</caption>"
    end

    if !isempty(header_row)
      cur_table *= "<thead><tr>"
      !isempty(header_col) && (cur_table *= "<th>")

      for cur_header in header_row
        cur_table *= "<th>$cur_header</th>"
      end

      cur_table *= "</tr></thead>"
    end

    cur_table *= "<tbody>"

    for ii in 1:size(cur_matrix, 1)

      cur_table *= "<tr>"
      if !isempty(header_col)
        cur_table *= "<td>"
        cur_table *= header_col[ii]
      end

      for jj in 1:size(cur_matrix, 2)
        cur_element = cur_matrix[ii, jj]

        if isa(cur_element, Dict)
          cur_table *= "<td class=\"$(cur_element["class"])\">"
          cur_value = cur_element["value"]
        else
          cur_table *= "<td>"
          cur_value = cur_element
        end

        cur_table *= string(cur_value)

        cur_table *= "</td>"
      end

      cur_table *= "</tr>"

    end

    cur_table *= "</tbody>"

    cur_table *= "</table>"

    return cur_table

  end


table(cur_matrix::Matrix; header_row=[], header_col=[], title=[], dp=Inf) =
    HTML(_table(cur_matrix; header_row=header_row, header_col=header_col,
    title=title, header=[], dp=dp))

function table(cur_matrix1::Matrix, cur_matrix2::Matrix; header_row=[], header_col=[],
    title=[], header=[], dp=Inf)
    title = isempty(title) ? ["", ""] : (title isa String ? repeat([title], 2) : title)
    t1 = _table(cur_matrix1; header_row=header_row, header_col=header_col, title=title[1],
            header=[], dp=dp)
    t2 = _table(cur_matrix2; header_row=header_row, title=title[2], header=[], dp=dp)

    head = !isempty(header) ? "<div><h3 align='center' style='padding: 10px'>$header</h2></div>" : ""
    HTML(head * "\n" * t1 * "\n" * t2)
end

function table(cur_matrix1::Array, cur_matrix2::Array; header_row=[], header_col=[], title=[],
    header=[], dp=Inf)
    if cur_matrix1[1] isa Real
        cur_matrix1 = reshape(cur_matrix1, 1, :)
    elseif cur_matrix1[1] isa Array
        cur_matrix1 = reduce(vcat, [reshape(x, 1, :) for x in cur_matrix1])
    end
    table(cur_matrix1; header_row=header_row, header_col=header_col, title=title,
        header=header, dp=dp)
end

function table(cur_matrix1::Array, cur_matrix2::Array; header_row=[], header_col=[],
    title=[], header=[], dp=Inf)
    if cur_matrix1[1] isa Real
        cur_matrix1 = reshape(cur_matrix1, 1, :)
    elseif cur_matrix1[1] isa Array
        cur_matrix1 = reduce(vcat, [reshape(x, 1, :) for x in cur_matrix1])
    end

    if cur_matrix2[1] isa Real
        cur_matrix2 = reshape(cur_matrix2, 1, :)
    elseif cur_matrix2[1] isa Array
        cur_matrix2 = reduce(vcat, [reshape(x, 1, :) for x in cur_matrix2])
    end
    table(cur_matrix1, cur_matrix2; header_row=header_row, header_col=header_col,
        title=title, header=header, dp=dp)
end
end
