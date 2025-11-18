class JSONBuilder:

    def assign_page_ranges(self, toc_json, total_pages):
        sections = toc_json["sections"]

        def assign(sections):
            for i in range(len(sections)):
                start = sections[i]["page_start"]
                
                if i < len(sections) - 1:
                    end = sections[i+1]["page_start"] - 1
                else:
                    end = total_pages

                sections[i]["page_end"] = end

                if sections[i]["children"]:
                    assign(sections[i]["children"])

        assign(sections)
        return toc_json

    def integrate_content(self, toc_struct, content_map):
        def attach(nodes):
            for node in nodes:
                key = (node["page_start"], node["page_end"], node["title"])
                node["parsed"] = content_map[key]
                if node["children"]:
                    attach(node["children"])

        attach(toc_struct["sections"])
        return toc_struct
